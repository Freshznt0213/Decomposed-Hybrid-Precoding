import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from random import shuffle
import time
import sys
import scipy
from utils import *
from MIMOHPC2 import MIMOHPCGNN

def cal_rate_hybrid_MS(H, WBB, WRF, SNR_dB, device):
    """
    MS:multi stream
    Ns:num of stream for one user
    算和数据率
    实部虚部  H(BS,2,K,Nr,Nt), WBB(BS,2,K,Ns,NRF), WRF(BS,2,1,NRF,Nt)
    """
    if H.dim() == 4:
        H = H.unsqueeze(-2)
    BS, _, K, Nr, Nt = H.shape
    Ns =  WBB.shape[3]
    NRF = WBB.shape[4]
    sigma2 = 10.0 ** (-SNR_dB / 10.0)
    Heq = complex_matmul(H,complex_conjT(WRF)) #[BS,2,K,Nr,NRF]
    Heq = torch.complex(Heq[:, 0], Heq[:, 1]).reshape(BS, K*Nr, NRF)  # BS,1,K*Nr,Nt
    WBB = torch.complex(WBB[:, 0], -WBB[:, 1]).reshape(BS,K*Ns,NRF)  # BS,1,K*Ns,Nt
    WBB = torch.transpose(WBB, -1, -2)  # BS,Nt,K*Ns
    HeqW = torch.matmul(Heq, WBB)  # BS,K(H)Nr,K(W)Ns
    HeqW = HeqW.reshape(BS, K, Nr, K, Ns).permute(0, 1, 3, 2, 4)  # BS,K(H),K(W),Nr,Ns
    HeqWh = HeqW.mH #for higher torch version, not for torch 1.9.1
    # HeqWh = HeqW.conj().transpose(-1,-2)
    Q = torch.matmul(HeqW, HeqWh)  # BS,K(H),K(W),Nr,Nr
    sigma2eye = sigma2 * torch.eye(Nr).to(device)
    S_mat = replicate_diagonal(torch.ones(Nr, Nr), K).reshape(1, K, Nr, K, Nr).permute(0, 1, 3, 2, 4).to(
        device)  # BS,K(H),K(W),Nr,Nr
    HeqW_mat = replicate_diagonal(torch.ones(Nr, Ns), K).reshape(1, K, Nr, K, Ns).permute(0, 1, 3, 2, 4).to(
        device)  # BS,K(H),K(W),Nr,Ns
    HeqW_k = torch.sum(HeqW * HeqW_mat, dim=2)  # BS,K(H),Nr,Ns
    S = Q * S_mat  # BS,K(H),K(W),Nr,Nr
    I = Q - S  # BS,K(H),K(W),Nr,Nr
    sumI = torch.sum(I, dim=2, keepdim=False)  # BS,K(H),Nr,Nr
    Ck = sumI + sigma2eye.unsqueeze(0).unsqueeze(0)  # BS,K(H),Nr,Nr
    Rk = torch.matmul((HeqW_k.mH), torch.linalg.solve(Ck, HeqW_k))  # BS,K(H),Ns,Ns
    rate = torch.sum(torch.real(torch.log2(torch.det(torch.eye(Ns).unsqueeze(0).unsqueeze(0).to(device) + Rk))))
    return rate / BS

class CHDataset(Dataset):
    def __init__(self, datafile,number,mode,init_aid):
        self.datafile = datafile
        self.init_aid = init_aid
        if mode == ".npy":
            self.H = np.load(self.datafile)
        else:
            raise ValueError("wrong mode")
        self.X = torch.from_numpy(self.H[0:number]).to(device).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.init_aid == None:
            return x
        else:
            raise TypeError('no such init_aid')

def create_data_loader(file, batch_size,number,shuffle,mode,init_aid):
    dataset = CHDataset(file,number,mode,init_aid)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


if __name__ == '__main__':
    time_start = time.time()

    K = 4
    Nt = 16
    Nr = 1
    Ns = 1
    NRF = 8
    Ncl = 4
    Nray = 5

    is_large = False
    if not is_large:
        SNR = 10
    else:
        BW = 100e6
        sigma2_dBm = -174 + 10 * torch.log10(torch.tensor(BW)) + 7
        sigma2_W = 10 ** ((sigma2_dBm-30)/10)
        SNR = -(sigma2_dBm - 30) - 10 #

    load_model = False
    load_learning_rate_from_saved_model = False

    # 信道矩阵数据集大小
    set1_number = 4000
    set2_number = 2000

    train_number = 1000
    test_number = 1000

    MISOmode = False
    channel_model = ['SV-4-5', ".npy"]

    trainfile = "./CHdata" + "/CH_K" + str(K) + "_Nt" + str(Nt) + \
                "_Nr" + str(Nr) + "_number" + str(set1_number) + "_" + channel_model[0] + channel_model[1]

    # trainfile = "./CHdata" + "/Dichasus007_Train_4User_16Ant.npy"

    testfile = "./CHdata" + "/CH_K" + str(K) + "_Nt" + str(Nt) + \
               "_Nr" + str(Nr) + "_number" + str(set2_number) + "_" + channel_model[0] + channel_model[1]

    # testfile = "./CHdata" + "/Dichasus007_Test_4User_16Ant.npy"


    BATCH_SIZE = 100
    testanhang = False  # 测试的时候是否等功率分配

    layer = [32,32,32,8]
    layer_PA = [8]*3
    layer_Analog = [16,16,16,16]

    MAX_EPOCH = 200
    device = torch.device("cuda:0")
    # device = torch.device("cpu")

    # 打开数据集
    print('K', K, 'Nt', Nt, 'Nr', Nr, 'Ncl', Ncl, 'Nray', Nray, 'SNR', SNR, 'train_number', train_number, 'testnumber',test_number)
    print('BatchSize', BATCH_SIZE, 'layer', layer)


    init_aid = None

    model_name = 'MIMOHPCGNN'
    model = MIMOHPCGNN(hidden_dimAnalog=layer_Analog,is_joint=True, hidden_dimPA=layer_PA, hidden_dimBF=layer,device=device)

    epoch2anhang = 10  # 有时候可能需要前几个epoch等功率分配，避免局部极小只服务1个用户
    model.to(device)
    print("Total number of paramerters:", sum(x.numel() for x in model.parameters() if x.requires_grad), epoch2anhang)
    print('model:', model_name, ' channel:', channel_model[0])

    if model_name == 'MIMOHPCGNN':
        optimizer = torch.optim.AdamW([
            {'params': model.Analog.parameters(),     'lr': 5e-2},
            {'params': model.Digital.BFmodule.parameters(), 'lr': 4e-2},
            {'params': model.Digital.PAmodule.parameters(), 'lr': 5e-2}
            # {'params': model.Digital.parameters(), 'lr': 8e-3}
        ], lr=0.01, weight_decay=0)
        # optimizer = torch.optim.Adam(model.parameters(),lr=0.01, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [60, 120, 180, 220, 360], gamma=0.3, last_epoch=-1)
    else:
        print('err')

    ##模型参数路径，各层网络参数+预编码算法+训练数据集大小
    model_path = './saved parameters/' + str(model_name) + '_' + '-'.join([str(i) for i in layer]) + '_n' + str(
        train_number)  + '_K' + str(K) + '_Nt' + str(Nt) + '_Nr' + str(Nr) + '_NRF' + str(NRF) + '_Ns' + str(Ns) + '_SNR' + str(SNR) + '_' + channel_model[0]

    model_path_load = './saved parameters/' + str(model_name) + '_' + '-'.join([str(i) for i in layer]) + '_n20000'+ '_K4' + '_Nt' + str(Nt) + '_Nr' + str(Nr) + '_NRF' + str(NRF) + '_Ns' + str(
        Ns) + '_SNR' + str(SNR) + '_' + channel_model[0]


    dr_train_list = []
    dr_test_list = []
    # sys.stdout.flush()

    if load_model:
        checkpoint = torch.load(model_path_load + "_epoch200")
        model.load_state_dict(checkpoint['net'])  # 将checkpoint中的 net 参数 传入 model
        optimizer.load_state_dict(checkpoint['optimizer'])
        # ###用于更新optimizer中的学习率
        if not load_learning_rate_from_saved_model:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-4
    trainloader = create_data_loader(file=trainfile, batch_size=BATCH_SIZE, number=train_number, shuffle=True, mode=channel_model[1], init_aid=init_aid)

    # trainloader1 = create_data_loader(file=trainfile, batch_size=BATCH_SIZE if BATCH_SIZE < test_number else test_number, number=test_number, shuffle=False, mode=channel_model[1],init_aid=init_aid)
    # testloader1 = create_data_loader(file=testfile, batch_size=BATCH_SIZE, number=test_number, shuffle=False, mode=channel_model[1],init_aid=init_aid)
    trainloader1 = create_data_loader(file=trainfile, batch_size=test_number, number=test_number, shuffle=False,mode=channel_model[1], init_aid=init_aid)
    testloader1 = create_data_loader(file=testfile, batch_size=test_number, number=test_number, shuffle=False,mode=channel_model[1], init_aid=init_aid)
    for epoch in range(MAX_EPOCH):
        model.train()
        for batch_idx, batch_x in enumerate(trainloader):
            if epoch < epoch2anhang:
                WRF_pred, WBB_pred = model(batch_x, NRF, Ns, anhang=True)
                # WRF_pred, WBB_pred = model(batch_x, anhang=True)
            else:
                WRF_pred, WBB_pred = model(batch_x, NRF, Ns, anhang=False)
                # WRF_pred, WBB_pred = model(batch_x, anhang=False)
            if isinstance(batch_x,list):
                batch_x = batch_x[0]
            loss = -cal_rate_hybrid_MS(H=batch_x, WBB=WBB_pred, WRF=WRF_pred, SNR_dB=SNR, device=device)

            optimizer.zero_grad()
            # model.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        if epoch % 1 == 0:
            btrain = len(trainloader1)
            btest = len(testloader1)
            model.eval()
            with torch.no_grad():
                sum_train_loss = 0
                sum_test_loss = 0
                for batch_idx, batch_x in enumerate(trainloader1):
                    WRF_pred, WBB_pred = model(batch_x, NRF, Ns, anhang=False)
                    # WRF_pred, WBB_pred = model(batch_x, anhang=False)
                    # time_end = time.time()
                    if isinstance(batch_x, list):
                        batch_x = batch_x[0]
                    loss = cal_rate_hybrid_MS(H=batch_x, WBB=WBB_pred, WRF=WRF_pred, SNR_dB=SNR, device=device)
                    sum_train_loss = sum_train_loss + loss.detach().cpu().numpy()

                for batch_idx, batch_x in enumerate(testloader1):
                    WRF_pred, WBB_pred = model(batch_x, NRF, Ns, anhang=False)
                    # WRF_pred, WBB_pred = model(batch_x, anhang=False)
                    if isinstance(batch_x, list):
                        batch_x = batch_x[0]
                    loss = cal_rate_hybrid_MS(H=batch_x, WBB=WBB_pred, WRF=WRF_pred, SNR_dB=SNR, device=device)
                    sum_test_loss = sum_test_loss + loss.detach().cpu().numpy()

                train_loss = sum_train_loss / btrain
                test_loss = sum_test_loss / btest

                print(epoch, train_loss, test_loss)
                print('*************************************************')
        if (epoch+1) % 100 == 0:
            state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(state, model_path + "_epoch" + str(epoch+1))

    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, model_path + "_epoch" + str(epoch+1))