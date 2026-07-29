from torch.utils.data import Dataset, DataLoader
from utils import *
from MIMOHPC2 import MIMOHPCGNN
import h5py

def cal_rate_hybrid_WB(H, WBB, WRF, SNR_dB, device):
    """
    WB:Wide_Band
    算和数据率
    实部虚部  H(BS,2,NRBG,K,Nt), WBB(BS,2,NRBG,K,NRF), WRF(BS,2,1,NRF,Nt)
    """
    BS, _, NRBG, K, Nt = H.size()
    sigma2 = 10.0 ** (-SNR_dB / 10.0)
    Weq = complex_matmul(WBB, WRF) #[BS,2,NRBG,K,Nt]
    Q = complex_matmul(H, complex_conjT(Weq))
    Q2 = ampli2(Q)
    D = torch.eye(K).to(device) * Q2
    sumD = torch.sum(D, dim=3, keepdim=True)
    sumQ = torch.sum(Q2, dim=3, keepdim=True)
    sinr = sumD / (sumQ - sumD + sigma2)
    rate = torch.sum(torch.log2(1.0 + sinr))
    return rate / BS / NRBG

def cal_rate_hybrid_WBMS(H, WBB, WRF, sigma2, device):
    """
    MS:multi stream
    Ns:num of stream for one user
    算和数据率
    实部虚部  H(BS,2,NRB,K,Nr,Nt), WBB(BS,2,NRB,K,Ns,NRF), WRF(BS,2,1,1,NRF,Nt)
    """
    BS, _, NRB, K, Nr, Nt = H.shape
    Ns =  WBB.shape[4]
    NRF = WBB.shape[5]
    Heq = complex_matmul(H,complex_conjT(WRF)) #[BS,2,NRB,K,Nr,NRF]
    Heq = torch.complex(Heq[:, 0], Heq[:, 1]).reshape(BS, NRB,K*Nr, NRF)  # BS,NRB,K*Nr,Nt
    WBB = torch.complex(WBB[:, 0], -WBB[:, 1]).reshape(BS,NRB,K*Ns,NRF)  # BS,NRB,K*Ns,Nt
    WBB = torch.transpose(WBB, -1, -2)
    HeqW = torch.matmul(Heq, WBB)  # BS,NRB,K(H)Nr,K(W)Ns
    HeqW = HeqW.reshape(BS, NRB, K, Nr, K, Ns).permute(0, 1, 2, 4, 3, 5)  # BS,NRB, K(H),K(W),Nr,Ns
    HeqWh = HeqW.mH #for higher torch version, not for torch 1.9.1
    # HeqWh = HeqW.conj().transpose(-1,-2)
    Q = torch.matmul(HeqW, HeqWh)  # BS,NRB, K(H),K(W),Nr,Nr
    sigma2eye = sigma2 * torch.eye(Nr).to(device)
    S_mat = replicate_diagonal(torch.ones(Nr, Nr), K).reshape(1, 1, K, Nr, K, Nr).permute(0, 1, 2, 4, 3, 5).to(
        device)  # BS,NRB,K(H),K(W),Nr,Nr
    HeqW_mat = replicate_diagonal(torch.ones(Nr, Ns), K).reshape(1, 1, K, Nr, K, Ns).permute(0, 1, 2, 4, 3, 5).to(
        device)  # BS,NRB,K(H),K(W),Nr,Ns
    HeqW_k = torch.sum(HeqW * HeqW_mat, dim=3)  # BS,NRB,K(H),Nr,Ns
    S = Q * S_mat  # BS,NRB,K(H),K(W),Nr,Nr
    I = Q - S  # BS,NRB,K(H),K(W),Nr,Nr
    sumI = torch.sum(I, dim=3, keepdim=False)  # BS,NRB,K(H),Nr,Nr
    Ck = sumI + sigma2eye.unsqueeze(0).unsqueeze(0).unsqueeze(0) # BS,NRB,K(H),Nr,Nr
    Rk = torch.matmul((HeqW_k.mH), torch.linalg.solve(Ck, HeqW_k))  # BS,NRB,K(H),Ns,Ns
    rate = torch.sum(torch.real(torch.log2(torch.det(torch.eye(Ns).unsqueeze(0).unsqueeze(0).to(device) + Rk))))
    return rate / BS / NRB

class CHDataset(Dataset):
    def __init__(self, datafile, number):
        self.datafile = datafile
        if not is_large_scale:
            self.H = np.load(self.datafile)
        else:
            # self.data = scipy.io.loadmat(self.datafile)
            # self.H = self.data['CH_all']
            with h5py.File(self.datafile, 'r') as f:
                self.H = np.array(f['CH_all']).transpose()

        self.X = torch.from_numpy(self.H[0:number]).to(device).float()#[number,2,NRBG,K,Nr,Nt]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        return x

def create_data_loader(file, batch_size,number,shuffle):
    dataset = CHDataset(file,number)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


if __name__ == '__main__':
    time_start = time.time()

    K = 8
    Nt = 1024
    Nr = 4
    Ns = 4
    NRF = 48
    NRBG = 16
    Ncl = 4
    Nray = 5

    max_num_sub = 3168

    is_large_scale = True

    if is_large_scale:
        fc = 28
        Pt_dBm = 35
        Pt_W = 10 ** ((Pt_dBm - 30) / 10)
        BW = 100e6
        sigma2_dBm = -174 + 10*torch.log10(torch.tensor(BW)) + 9
        sigma2_W = 10 ** ((sigma2_dBm - 30) / 10)
    else:
        SNR = 0
        Pt_W = 1
        sigma2_W = 10.0 ** (-SNR / 10.0)

    load_model = True
    load_learning_rate_from_saved_model = True

    set1_number = 400
    set2_number = 100

    train_number = 400
    test_number = 100

    if not is_large_scale:
        trainfile = "./Dataset" + "/Train_WBSV" + "_NRBG" + str(NRBG) + "_K" + str(K) \
                    + "_Nt" + str(Nt) + "_Nr" + str(Nr) + "_Ncl" + str(Ncl) + "_Nray" + str(Nray) \
                    + "/H_tensor.npy"


        testfile = "./Dataset" + "/Test_WBSV" + "_NRBG" + str(NRBG) + "_K" + str(K) \
                    + "_Nt" + str(Nt) + "_Nr" + str(Nr) + "_Ncl" + str(Ncl) + "_Nray" + str(Nray) \
                    + "/H_tensor.npy"
    else:
        trainfile = "./Dataset" + "/Train_WBUma" + "_fc" + str(fc) + "_NRBG" + str(NRBG) + "_K" + str(K) \
                    + "_Nt" + str(Nt) + "_Nr" + str(Nr) + "_num" + str(set1_number) \
                    + "/H_tensor.mat"

        testfile = "./Dataset" + "/Test_WBUma" + "_fc" + str(fc) + "_NRBG" + str(NRBG) + "_K" + str(K) \
                   + "_Nt" + str(Nt) + "_Nr" + str(Nr) + "_num" + str(set2_number) \
                   + "/H_tensor.mat"

    BATCH_SIZE = 20
    testanhang = False  # 测试的时候是否等功率分配

    layer = [32,32,32,8]
    layer_PA = [8]*3
    layer_Analog = [16,16,16,16]
    layer_init = [8,8,8]

    MAX_EPOCH = 540
    device = torch.device("cuda:0")
    # device = torch.device("cpu")

    # 打开数据集
    print('K', K, 'Nt', Nt, 'Nr', Nr, 'Ncl', Ncl, 'Nray', Nray, 'train_number', set1_number, 'testnumber',set2_number)
    print('BatchSize', BATCH_SIZE, 'layer', layer)


    init_aid = None

    # Sequencial Hybrid MIMO GNN
    model_name = 'MIMOHPCGNN'
    model = MIMOHPCGNN(hidden_dimAnalog=layer_Analog,is_joint=True, hidden_dimPA=layer_PA, hidden_dimBF=layer,device=device)

    epoch2anhang = 0  # 有时候可能需要前几个epoch等功率分配，避免局部极小只服务1个用户
    model.to(device)
    print("Total number of paramerters:", sum(x.numel() for x in model.parameters() if x.requires_grad), epoch2anhang)

    if model_name == 'MIMOHPCGNN':
        optimizer = torch.optim.AdamW([
            {'params': model.Analog.parameters(),     'lr': 1e-2},
            {'params': model.Digital.BFmodule.parameters(), 'lr': 1e-3},
            {'params': model.Digital.PAmodule.parameters(), 'lr': 5e-4}
        ], lr=0.01, weight_decay=0)
        # optimizer = torch.optim.Adam(model.parameters(),lr=0.01, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [60, 120, 180, 220, 360], gamma=0.3, last_epoch=-1)
    else:
        print('err')

    ##模型参数路径，各层网络参数+预编码算法+训练数据集大小
    model_path = './saved parameters/' + 'fading_' + str(is_large_scale) + '_' + str(model_name) + '_' + '-'.join([str(i) for i in layer]) + '_n' + str(
        set1_number) + '_NRB' + str(NRBG) + '_K' + str(K) + '_Nt' + str(Nt) + '_Nr' + str(Nr) + '_NRF' + str(NRF)

    ##模型参数路径，各层网络参数+预编码算法+训练数据集大小
    model_path_load = './saved parameters/' + 'fading_' + str(is_large_scale) + '_' + str(model_name) + '_' + '-'.join([str(i) for i in layer]) + '_n' + str(
        400) + '_NRB' + str(16) + '_K' + str(8) + '_Nt' + str(1024) + '_Nr' + str(4) + '_NRF' + str(48)

    dr_train_list = []
    dr_test_list = []
    # sys.stdout.flush()

    if load_model:
        checkpoint = torch.load(model_path_load + "_epoch407")
        model.load_state_dict(checkpoint['net'])  # 将checkpoint中的 net 参数 传入 model
        optimizer.load_state_dict(checkpoint['optimizer'])
        # ###用于更新optimizer中的学习率
        if not load_learning_rate_from_saved_model:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-5
    trainloader = create_data_loader(file=trainfile, batch_size=BATCH_SIZE, number=train_number, shuffle=True)

    # trainloader1 = create_data_loader(file=trainfile, batch_size=BATCH_SIZE, number=set2_number, shuffle=False)
    testloader1 = create_data_loader(file=testfile, batch_size=BATCH_SIZE, number=test_number, shuffle=False)
    # trainloader1 = create_data_loader(file=trainfile, batch_size=test_number, number=test_number, shuffle=False,mode=channel_model[1], init_aid=init_aid)
    # testloader1 = create_data_loader(file=testfile, batch_size=1, number=test_number, shuffle=False,mode=channel_model[1], init_aid=init_aid)
    traintimesum = 0
    for epoch in range(MAX_EPOCH):
        model.train()
        train_starttime = time.time()
        for batch_idx, batch_x in enumerate(trainloader):
            if epoch < epoch2anhang:
                WRF_pred, WBB_pred = model(batch_x, NRF, Ns, Pt_W, sigma2_W, max_num_sub, is_large_scale=is_large_scale, anhang=True)
            else:
                WRF_pred, WBB_pred = model(batch_x, NRF, Ns, Pt_W, sigma2_W, max_num_sub, is_large_scale=is_large_scale, anhang=False)
            if isinstance(batch_x,list):
                batch_x = batch_x[0]
            loss = -cal_rate_hybrid_WBMS(H=batch_x, WBB=WBB_pred, WRF=WRF_pred, sigma2=sigma2_W, device=device)

            optimizer.zero_grad()
            # model.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        train_endtime = time.time()
        traintimesum = traintimesum + train_endtime - train_starttime
        if epoch % 1 == 0:
            # btrain = len(trainloader1)
            btest = len(testloader1)
            model.eval()
            with torch.no_grad():
                sum_train_loss = 0
                sum_test_loss = 0
                sum_test_time = 0
                # for batch_idx, batch_x in enumerate(trainloader1):
                #     WRF_pred, WBB_pred = model(batch_x, NRF, Ns, Pt_W, sigma2_W, max_num_sub, is_large_scale=is_large_scale, anhang=False)
                #     # WRF_pred, WBB_pred = model(batch_x, anhang=True)
                #     # time_end = time.time()
                #     if isinstance(batch_x, list):
                #         batch_x = batch_x[0]
                #     loss = cal_rate_hybrid_WBMS(H=batch_x, WBB=WBB_pred, WRF=WRF_pred, sigma2=sigma2_W, device=device)
                #     sum_train_loss = sum_train_loss + loss.detach().cpu().numpy()

                for batch_idx, batch_x in enumerate(testloader1):
                    time_start = time.time()
                    WRF_pred, WBB_pred = model(batch_x, NRF, Ns, Pt_W, sigma2_W, max_num_sub, is_large_scale=is_large_scale, anhang=False)
                    time_end = time.time()
                    # WRF_pred, WBB_pred = model(batch_x, anhang=True)
                    if isinstance(batch_x, list):
                        batch_x = batch_x[0]
                    loss = cal_rate_hybrid_WBMS(H=batch_x, WBB=WBB_pred, WRF=WRF_pred, sigma2=sigma2_W, device=device)
                    # if loss.isnan().any():
                    #     btest -= 1
                    # else:
                    sum_test_loss = sum_test_loss + loss.detach().cpu().numpy()
                    sum_test_time = sum_test_time + time_end - time_start

                # train_loss = sum_train_loss / btrain
                test_loss = sum_test_loss / btest

                test_time = sum_test_time / set2_number
                print(epoch, 1, test_loss)
                # print(epoch, train_loss, test_loss)
                print(traintimesum)
                # print(test_time)
                print('*************************************************')
                dr_test_list.append(test_loss)
                # dr_train_list.append(train_loss)
                # sys.stdout.flush()
                # time_start = time.time()
        if (epoch+1) % 100 == 0:
            state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(state, model_path + "_epoch" + str(epoch+1))

    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, model_path + "_epoch" + str(epoch+1))