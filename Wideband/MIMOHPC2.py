import torch
import torch.nn as nn
from utils import *
from SepGNN import BFModule, PAModule,SepGNN
import scipy

def CMA(x):
    """
    激活函数：
    1. 将输入张量重塑为 [BS, C/2, K, Nt, 2]
    2. 按最后一个维度求模
    3. 用输入除以模，归一化最后一个维度的值

    Args:
        x (torch.Tensor): 输入张量，形状为 [BS, C, K, Nt]
    Returns:
        torch.Tensor: 输出张量，形状与输入相同
    """
    # 检查输入是否符合预期
    assert x.shape[1] % 2 == 0, "C 必须是偶数"

    # 重塑为 [BS, C/2, K, Nt, 2]
    x0 = x.view(x.shape[0], x.shape[1] // 2, x.shape[2], x.shape[3], 2)

    # 按最后一个维度求模
    mod = torch.sqrt((x0**2).sum(dim=-1, keepdim=True))  # 保持维度以便后续操作

    # 避免除以零，将模中的零替换为一个小值
    mod = torch.clamp(mod, min=1e-8)

    # 用输入除以模，归一化
    normalized_x = x0 / mod

    # 返回与输入形状一致的张量
    return normalized_x.view_as(x)

def simplerank(H):
    # H: 输入复数张量，形状为 [Batchsize, 2, NRBG, K, Nt]
    # 返回：形状为 [Batchsize, K, K] 的置换矩阵

    H = torch.complex(H[:,0],H[:,1])
    #[BS,NRBG,K,Nt]
    # Step 1: 计算每个样本的行向量 L2 范数
    norms = torch.norm(H, dim=3)  # [Batchsize, K]，每行向量的 L2 范数

    # Step 2: 对 norms 排序，得到排序后的索引
    _, sorted_indices = torch.sort(norms, dim=2)  # [Batchsize, NRBG, K]，按 norm 从小到大的索引

    # Step 3: 生成置换矩阵
    # 创建一个初始矩阵，形状为 [Batchsize, K, K]，用于存储置换矩阵
    PEmat = torch.zeros(H.size(0), H.size(1), H.size(2), H.size(2), dtype=torch.float32, device=H.device)

    # 使用 gather 和 one-hot 编码生成置换矩阵
    PEmat.scatter_(3, sorted_indices.unsqueeze(3), 1.0)
    return PEmat

def cal_ch_Heq(H0,WRF,device):
    '''
    used for sequencial structrue
    WRF = QR
    unitary_mat:Q
    upper triangular:R
    chol_psuedo:R^{-1}
    :return:
    '''
    #H0:[BS,2,K,Nt], WRF:[BS,2,NRF,Nt]
    BS, _, NRBG, K, Nt = H0.size()
    NRF = WRF.size(2)
    WRF = WRF.unsqueeze(2) #[BS,2,1,NRF,Nt]
    WRF_comp = torch.complex(WRF[:,0],-WRF[:,1]).transpose(-1,-2) #[BS,Nt,NRF]
    [WRF_Q,WRF_R] = torch.linalg.qr(WRF_comp) #WRF_R:[BS,NRF,NRF]
    # [WRF_Q, WRF_R] = manual_qr(WRF_comp)  # WRF_R:[BS,NRF,NRF]

    chol_psuedo = torch.linalg.pinv(WRF_R) #[BS,NRF,NRF]
    # chol_psuedo = torch.linalg.solve(WRF_R,psuedo_aid)
    chol_psuedo = torch.cat([torch.real(chol_psuedo).unsqueeze(1),torch.imag(chol_psuedo).unsqueeze(1)],dim=1)

    unitary_mat = torch.cat([torch.real(WRF_Q).unsqueeze(1), torch.imag(WRF_Q).unsqueeze(1)], dim=1)

    ch_eq = complex_matmul(H0,unitary_mat)
    return ch_eq, chol_psuedo

def cal_svd_Heq(H0,WRF,device):
    '''
    used for sequencial structrue
    WRF = U*D*Vh
    semi_unitary_mat:U*Vh
    half_psuedo = V*D^{-1}*Vh

    :return:
    '''
    #H0:[BS,2,NRBG,K,Nt], WRF:[BS,2,NRF,Nt]
    BS, _, NRBG, K, Nt = H0.size()
    NRF = WRF.size(2)
    WRF = WRF.unsqueeze(2) #[BS,2,1,NRF,Nt]
    WRF_comp = torch.complex(WRF[:,0],-WRF[:,1]).transpose(-1,-2) #[BS,1,Nt,NRF]
    WRF_U,WRF_D,WRF_Vh = torch.linalg.svd(WRF_comp) # U:[BS,1,Nt,Nt],sigma:[BS,1,NRF],Vh:[BS,1,NRF,NRF]
    WRF_D_inv = torch.diag_embed(1 / WRF_D).to(torch.complex64)
    semi_unitary_mat = torch.matmul(WRF_U[:,:,:,0:NRF],WRF_Vh)
    semi_unitary_mat = torch.cat([torch.real(semi_unitary_mat).unsqueeze(1), torch.imag(semi_unitary_mat).unsqueeze(1)], dim=1)
    half_inv = torch.matmul(torch.matmul(WRF_Vh.mH,WRF_D_inv),WRF_Vh)
    half_inv = torch.cat([torch.real(half_inv).unsqueeze(1), torch.imag(half_inv).unsqueeze(1)], dim=1)
    ch_eq = complex_matmul(H0, semi_unitary_mat)
    return ch_eq, half_inv

def cal_ch_What(WBB,WRF,device):
    '''
    used for sequencial structrue
    WRF = QR
    semi_unitary_mat:Q
    upper triangular:R
    chol_psuedo:R^{-1}
    :return:
    '''
    #WBB:[BS,2,K,Nt], WRF:[BS,2,NRF,Nt]
    BS, _, K, Nt = WBB.size()
    NRF = WRF.size(2)
    WRF_comp = torch.complex(WRF[:,0],-WRF[:,1]).transpose(-1,-2) #[BS,Nt,NRF]
    [WRF_Q,WRF_R] = torch.linalg.qr(WRF_comp) #WRF_R:[BS,NRF,NRF]
    semi_unitary_mat = torch.cat([torch.real(WRF_Q).unsqueeze(1), torch.imag(WRF_Q).unsqueeze(1)], dim=1)
    WBBhat= complex_matmul(WBB,semi_unitary_mat)
    chol_inv = torch.linalg.pinv(WRF_R) #[BS,NRF,NRF]
    chol_inv = torch.cat([torch.real(chol_inv).unsqueeze(1),torch.imag(chol_inv).unsqueeze(1)],dim=1)
    return WBBhat, chol_inv

def cal_svd_What(WBB,WRF,device):
    '''
    used for sequencial structrue
    WRF = U*D*Vh
    semi_unitary_mat:U*Vh
    half_psuedo = V*D^{-1}*Vh
    :return:
    '''
    #WBB:[BS,2,K,Nt], WRF:[BS,2,NRF,Nt]
    BS, _, K, Nt = WBB.size()
    NRF = WRF.size(2)
    WRF_comp = torch.complex(WRF[:,0],-WRF[:,1]).transpose(-1,-2) #[BS,Nt,NRF]
    WRF_U,WRF_D,WRF_Vh = torch.linalg.svd(WRF_comp) # U:[BS,Nt,Nt],sigma:[BS,NRF],Vh:[BS,NRF,NRF]
    WRF_D_inv = torch.diag_embed(1 / WRF_D).to(torch.complex64)
    semi_unitary_mat = torch.matmul(WRF_U[:,:,0:NRF],WRF_Vh)
    semi_unitary_mat = torch.cat([torch.real(semi_unitary_mat).unsqueeze(1), torch.imag(semi_unitary_mat).unsqueeze(1)], dim=1)
    WBBhat = complex_matmul(WBB, semi_unitary_mat)
    half_inv = torch.matmul(torch.matmul(WRF_Vh.mH,WRF_D_inv),WRF_Vh)
    half_inv = torch.cat([torch.real(half_inv).unsqueeze(1), torch.imag(half_inv).unsqueeze(1)], dim=1)
    return WBBhat, half_inv

class multipliernet(nn.Module):
    """
    2D-GNN with attention
    """
    def __init__(self, hidden_dim, Ns, device):
        super(multipliernet, self).__init__()
        self.P1 = nn.ParameterList()
        self.P2 = nn.ParameterList()
        self.P3 = nn.ParameterList()
        self.P4 = nn.ParameterList()

        self.Ns = Ns
        self.device = device
        self.batch_norms = torch.nn.ModuleList()

        self.activation = nn.ReLU()

        self.dim = [2] + list(hidden_dim) + [1]
        for i in range(len(self.dim) - 2):
            self.batch_norms.append(nn.BatchNorm2d(self.dim[i + 1]))
        for i in range(len(self.dim) - 1):
            ini = 1.0 / torch.sqrt(torch.FloatTensor([self.dim[i + 1] * self.dim[i]]))
            self.P1.append(nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
            self.P2.append(nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
            self.P3.append(nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
            self.P4.append(nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))


    def forward(self, H0, device):
        BS, _, KS, Nt = H0.size()
        K = KS // self.Ns
        A = H0.contiguous()

        for i in range(len(self.P1)):
            A1 = torch.matmul(self.P1[i], A.view([BS, self.dim[i], -1])).view([BS, self.dim[i + 1], KS, Nt])
            A2 = torch.matmul(self.P2[i], torch.mean(A, -1).view(BS, self.dim[i], -1)).view(BS,self.dim[i + 1],KS, 1)
            A3 = torch.matmul(self.P3[i], torch.mean(A, -2).view(BS, self.dim[i], -1)).view(BS,self.dim[i + 1], 1, Nt)
            A4 = torch.matmul(self.P4[i], torch.mean(A.view(BS, self.dim[i], K, self.Ns, Nt),-2,keepdim=True).repeat(1,1,1,self.Ns,1).view(BS,2*self.dim[i],-1)).view(BS, self.dim[i + 1], KS, Nt)

            A = A1 + 0.1 * A2 + 0.1 * A3 + 0.2 * A4
            # 激活
            if i != len(self.P1) - 1:
                # A = self.activation(A)
                A = self.batch_norms[i](A)
                A = self.activation(A)

        # A:[BS,1,KS,Nt]
        output = A.reshape(BS,K,self.Ns,Nt)
        output = torch.mean(output,dim=[-1,-2])

        return output #[BS,K]

class AAGNN(nn.Module):
    """
    2D-GNN with attention
    """
    def __init__(self, hidden_dim, NRF, device):
        super(AAGNN, self).__init__()
        self.P1 = nn.ParameterList()
        self.P2 = nn.ParameterList()
        self.P3 = nn.ParameterList()
        # self.bias = nn.ParameterList()

        self.Q = nn.ParameterList()
        self.Key = nn.ParameterList()
        self.V = nn.ParameterList()

        self.NRF = NRF
        self.device = device
        self.batch_norms = torch.nn.ModuleList()

        self.activation = nn.LeakyReLU()
        # self.activation = nn.ReLU()
        # self.activation = nn.Softplus()
        # self.activation = nn.Tanh()
        # self.activation = normuni2

        self.dim = [2] + list(hidden_dim) + [2]
        for i in range(len(self.dim) - 2):
            self.batch_norms.append(nn.BatchNorm2d(self.dim[i + 1]))
        for i in range(len(self.dim) - 1):
            ini = 1.0 / torch.sqrt(torch.FloatTensor([self.dim[i + 1] * self.dim[i]]))
            self.P1.append(nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
            self.P2.append(nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
            self.P3.append(nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
            self.Q.append(nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
            self.Key.append(nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
            self.V.append(nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
            #self.bias.append(nn.Parameter(torch.rand([1,self.dim[i + 1],1,1], requires_grad=True) * 2 * ini - ini))

    def forward(self, H0, device):
        BS, _, K, Nt = H0.size()
        PEmat = simplerank(H0).unsqueeze(1)*torch.ones([1, 2, 1, 1]).to(self.device) #[BS,1,K,K],right matmul
        F = dft_matrix(self.NRF)[:,:,0:K].unsqueeze(0).to(device)
        F = complex_matmul(F,PEmat)
        WRF0 = complex_matmul(F,H0)/K
        A = WRF0
        A = A.contiguous()

        for i in range(len(self.P1)):
            A1 = torch.matmul(self.P1[i], A.view([BS, self.dim[i], -1])).view([BS, self.dim[i + 1], self.NRF, Nt])
            A2 = torch.matmul(self.P2[i], torch.mean(A, -1).view(BS, self.dim[i], -1)).view(BS,self.dim[i + 1],self.NRF, 1)
            A3 = torch.matmul(self.P3[i], torch.mean(A, -2).view(BS, self.dim[i], -1)).view(BS,self.dim[i + 1], 1, Nt)

            # attention
            q = torch.matmul(self.Q[i], A.view([BS, self.dim[i], -1])).view([BS, self.dim[i + 1], self.NRF, Nt])
            k = torch.matmul(self.Key[i], A.view([BS, self.dim[i], -1])).view([BS, self.dim[i + 1], self.NRF, Nt]).transpose(-1, -2)

            alpha = nn.Tanh()(torch.matmul(q, k) / Nt) #[BS,_,K,K]

            v = torch.matmul(self.V[i], A.view(BS, self.dim[i], -1)).view(BS,self.dim[i + 1],self.NRF, Nt)
            A4 = torch.matmul(alpha, v).view([BS, self.dim[i + 1], self.NRF, Nt]) / self.NRF

            A = A1 + 0.1 * A2 + 0.1 * A3 + 0.2 * A4# + 0.1*self.bias[i]
            # 激活
            if i != len(self.P1) - 1:
                # A = self.activation(A)
                A = self.batch_norms[i](A)
                A = self.activation(A)

        # A:[BS,2,NRF,Nt]
        WRF_norm = torch.norm(A, dim=1, keepdim=True)
        WRF = A / WRF_norm
        return WRF

class EdgeGNN(nn.Module):
    """
    2D vanilla GNN
    """
    def __init__(self, hidden_dim, device):
        super(EdgeGNN, self).__init__()
        self.P1 = nn.ParameterList()
        self.P2 = nn.ParameterList()
        self.P3 = nn.ParameterList()

        self.device = device
        self.batch_norms = torch.nn.ModuleList()

        self.activation = nn.LeakyReLU()
        # self.activation = nn.ReLU()
        # self.activation = nn.Softplus()
        # self.activation = nn.Tanh()
        # self.activation = normuni2

        self.dim = [2] + list(hidden_dim) + [2]
        for i in range(len(self.dim) - 2):
            self.batch_norms.append(nn.BatchNorm2d(self.dim[i + 1]))
        for i in range(len(self.dim) - 1):
            ini = 1.0 / torch.sqrt(torch.FloatTensor([self.dim[i + 1] * self.dim[i]]))
            self.P1.append(nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
            self.P2.append(nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
            self.P3.append(nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
            #self.bias.append(nn.Parameter(torch.rand([1,self.dim[i + 1],1,1], requires_grad=True) * 2 * ini - ini))

    def forward(self, H0, NRF):
        BS, _, NRBG, K, Nt = H0.size()
        if True:
            PEmat = simplerank(H0).unsqueeze(1)*torch.ones([1, 2, 1, 1, 1]).to(self.device) #[BS,1,NRBG, K,K],right matmul
            F = dft_matrix(NRF)[:,:,0:K].unsqueeze(0).unsqueeze(2).to(self.device)#.flip(dims=[-1])
            F = complex_matmul(F,PEmat)
            WRF0 = complex_matmul(F,H0)
            WRF0 = torch.mean(WRF0,dim=2)
        else:
            idx = torch.linspace(0, Nt - 1, NRF).round().long()
            Codebook_init = dft_matrix(Nt)[:,:,idx].unsqueeze(0).to(self.device)
            WRF0 = complex_conjT(Codebook_init)/ torch.sqrt(torch.tensor(Nt, dtype=torch.float32).to(self.device))
            WRF0 = WRF0.repeat([BS,1,1,1])
        A = WRF0
        A = A.contiguous() #[BS,2,NRF,Nt]

        for i in range(len(self.P1)):
            A1 = torch.matmul(self.P1[i], A.view([BS, self.dim[i], -1])).view([BS, self.dim[i + 1], NRF, Nt])
            # A2 = torch.matmul(self.P2[i], torch.mean(A, -1).view(BS, self.dim[i], -1)).view(BS,self.dim[i + 1],NRF, 1)
            A3 = torch.matmul(self.P3[i], torch.mean(A, -2).view(BS, self.dim[i], -1)).view(BS,self.dim[i + 1], 1, Nt)


            A = A1 + 0.1 * A3
            # 激活
            if i != len(self.P1) - 1:
                A = self.activation(A)
                A = self.batch_norms[i](A)
                # A = self.activation(A)

        # A:[BS,2,NRF,Nt]
        WRF_norm = torch.norm(A, dim=1, keepdim=True)
        WRF = A / WRF_norm / torch.sqrt(torch.tensor(Nt, dtype=torch.float32).to(self.device))
        return WRF

class EdgeGRNN(nn.Module):
    """
    2D vanilla GNN with H residual(a model induced),dont consider M^{-2}
    """
    def __init__(self, hidden_dim, device):
        super(EdgeGRNN, self).__init__()
        self.P1 = nn.ParameterList()
        self.P2 = nn.ParameterList()
        self.P3 = nn.ParameterList()

        self.device = device
        self.batch_norms = torch.nn.ModuleList()

        self.activation = nn.LeakyReLU()

        self.dim = [2] + list(hidden_dim) + [2]
        for i in range(len(self.dim) - 2):
            self.batch_norms.append(nn.BatchNorm2d(self.dim[i + 1]))
        for i in range(len(self.dim) - 1):
            ini = 1.0 / torch.sqrt(torch.FloatTensor([self.dim[i + 1] * self.dim[i]]))
            self.P1.append(nn.Parameter(torch.rand([self.dim[i + 1], 2*self.dim[i]], requires_grad=True) * 2 * ini - ini))
            self.P2.append(nn.Parameter(torch.rand([self.dim[i + 1], 2*self.dim[i]], requires_grad=True) * 2 * ini - ini))
            self.P3.append(nn.Parameter(torch.rand([self.dim[i + 1], 2*self.dim[i]], requires_grad=True) * 2 * ini - ini))
            #self.bias.append(nn.Parameter(torch.rand([1,self.dim[i + 1],1,1], requires_grad=True) * 2 * ini - ini))

    def forward(self, H0, NRF):
        #H0:[BS,2,KNs,Nt]
        BS, _, K, Nt = H0.size()
        PEmat = simplerank(H0).unsqueeze(1)*torch.ones([1, 2, 1, 1]).to(self.device) #[BS,1,K,K],right matmul
        F = dft_matrix(NRF)[:,:,0:K].unsqueeze(0).to(self.device)#.flip(dims=[-1])
        F = complex_matmul(F,PEmat)
        WRF0 = complex_matmul(F,H0)
        A = WRF0
        A = A.contiguous() #[BS,2,NRF,Nt]

        H0_Re = H0.permute(0, 1, 3, 2)[:,0:1,:,:]; H0_Im = -1 * H0.permute(0, 1, 3, 2)[:,1:2,:,:] #####conjugate
        H0h = torch.concat((H0_Re, H0_Im), dim=1)####conjugate (Batch_size, 2, Nt, K)
        H0hH0 = complex_matmul(H0h,H0).permute(0,2,3,1).unsqueeze(1)/Nt/K #[BS,1,Nt,Nt,2]
        for i in range(len(self.P1)):
            A = A.view(BS, int(A.shape[1]/2), 2, NRF, Nt).permute(0,1,3,4,2)
            ####H0h:(Batch_size, 1, Nt, K, 2) ; D:(Batch_size, channel/2, K, Nt, 2)
            B = complex_matmul_model(A, H0hH0 * torch.ones([1, int(A.shape[1]), 1, 1, 1]).to(self.device))
            C = torch.concat((A, B), dim=-1)  ### dim(Batch_size, channel/2, K, Nt, 4)
            C = torch.reshape(C.permute(0, 1, 4, 2, 3), [-1, int(C.shape[1] * C.shape[-1]), NRF, Nt])
            A1 = torch.matmul(self.P1[i], C.view([BS, 2*self.dim[i], -1])).view([BS, self.dim[i + 1], NRF, Nt])
            A2 = torch.matmul(self.P2[i], torch.mean(C, -1).view(BS, 2*self.dim[i], -1)).view(BS,self.dim[i + 1],NRF, 1)
            A3 = torch.matmul(self.P3[i], torch.mean(C, -2).view(BS, 2*self.dim[i], -1)).view(BS,self.dim[i + 1], 1, Nt)


            A = A1 + 0.1 * A2 + 0.1 * A3
            # 激活
            if i != len(self.P1) - 1:
                # A = self.activation(A)
                A = self.batch_norms[i](A)
                A = self.activation(A)

        # A:[BS,2,NRF,Nt]
        WRF_norm = torch.norm(A, dim=1, keepdim=True)
        WRF = A / WRF_norm / torch.sqrt(torch.tensor(Nt, dtype=torch.float32).to(self.device))
        return WRF

class EdgeGRNN_simp(nn.Module):
    """
    2D vanilla GNN with H residual(a model induced),dont consider M^{-2}
    """
    def __init__(self, hidden_dim, device):
        super(EdgeGRNN_simp, self).__init__()
        self.P1 = nn.ParameterList()
        # self.P2 = nn.ParameterList()
        # self.P3 = nn.ParameterList()
        self.P4 = nn.ParameterList()

        self.device = device
        self.batch_norms = torch.nn.ModuleList()

        self.activation = nn.LeakyReLU()

        self.dim = [2] + list(hidden_dim) + [2]
        for i in range(len(self.dim) - 2):
            self.batch_norms.append(nn.BatchNorm2d(self.dim[i + 1]))
        for i in range(len(self.dim) - 1):
            ini = 1.0 / torch.sqrt(torch.FloatTensor([self.dim[i + 1] * self.dim[i]]))
            self.P1.append(nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
            # self.P2.append(nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
            # self.P3.append(nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
            self.P4.append(nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
            #self.bias.append(nn.Parameter(torch.rand([1,self.dim[i + 1],1,1], requires_grad=True) * 2 * ini - ini))

    def forward(self, H0, NRF):
        #H0:[BS,2,NRBG, K,Nt]
        BS, _, NRBG, K, Nt = H0.size()
        if True:
            PEmat = simplerank(H0).unsqueeze(1)*torch.ones([1, 2, 1, 1, 1]).to(self.device) #[BS,1,K,K],right matmul
            F = dft_matrix(NRF)[:,:,0:K].unsqueeze(0).unsqueeze(2).to(self.device)#.flip(dims=[-1])
            F = complex_matmul(F,PEmat)
            WRF0 = torch.mean(complex_matmul(F,H0),dim=2)
        else:
            idx = torch.linspace(0, Nt - 1, NRF).round().long()
            Codebook_init = dft_matrix(Nt)[:,:,idx].unsqueeze(0).to(self.device)
            WRF0 = complex_conjT(Codebook_init)/ torch.sqrt(torch.tensor(Nt, dtype=torch.float32).to(self.device))
            WRF0 = WRF0.repeat([BS,1,1,1])
        A = WRF0
        A = A.contiguous() #[BS,2,NRF,Nt]

        H0 = torch.reshape(H0, [BS,_,NRBG*K,Nt])
        H0_Re = H0.permute(0, 1, 3, 2)[:,0:1,:,:]; H0_Im = -1 * H0.permute(0, 1, 3, 2)[:,1:2,:,:] #####conjugate
        H0h = torch.concat((H0_Re, H0_Im), dim=1)####conjugate (Batch_size, 2, Nt, K)
        H0hH0 = complex_matmul(H0h,H0).permute(0,2,3,1).unsqueeze(1)/Nt #[BS,1,Nt,Nt,2]
        for i in range(len(self.P1)):
            A = A.view(BS, int(A.shape[1]/2), 2, NRF, Nt).permute(0,1,3,4,2)
            ####H0h:(Batch_size, 1, Nt, K, 2) ; D:(Batch_size, channel/2, K, Nt, 2)
            B = complex_matmul_model(A, H0hH0 * torch.ones([1, int(A.shape[1]), 1, 1, 1]).to(self.device))
            A = torch.reshape(A.permute(0, 1, 4, 2, 3), [-1, int(A.shape[1] * A.shape[-1]), NRF, Nt])
            B = torch.reshape(B.permute(0, 1, 4, 2, 3), [-1, int(B.shape[1] * B.shape[-1]), NRF, Nt])
            # C = torch.concat((A, B), dim=-1)  ### dim(Batch_size, channel/2, K, Nt, 4)
            # C = torch.reshape(C.permute(0, 1, 4, 2, 3), [-1, int(C.shape[1] * C.shape[-1]), NRF, Nt])
            A1 = torch.matmul(self.P1[i], A.view([BS, self.dim[i], -1])).view([BS, self.dim[i + 1], NRF, Nt])
            # A2 = torch.matmul(self.P2[i], torch.mean(A, -1).view(BS, self.dim[i], -1)).view(BS,self.dim[i + 1],NRF, 1)
            # A3 = torch.matmul(self.P3[i], torch.mean(A, -2).view(BS, self.dim[i], -1)).view(BS,self.dim[i + 1], 1, Nt)
            A4 = torch.matmul(self.P4[i], B.view([BS, self.dim[i], -1])).view([BS, self.dim[i + 1], NRF, Nt])

            A = A1 + A4
            # 激活
            if i != len(self.P1) - 1:
                # A = self.activation(A)
                A = self.batch_norms[i](A)
                A = self.activation(A)

        # A:[BS,2,NRF,Nt]
        WRF_norm = torch.norm(A, dim=1, keepdim=True)
        WRF = A / WRF_norm / torch.sqrt(torch.tensor(Nt, dtype=torch.float32).to(self.device))
        return WRF

class EdgeGRNN_simp_softmax(nn.Module):
    """
    2D vanilla GNN with H residual(a model induced),dont consider M^{-2}
    """
    def __init__(self, hidden_dim, device):
        super(EdgeGRNN_simp_softmax, self).__init__()
        self.P1 = nn.ParameterList()
        self.P2 = nn.ParameterList()
        self.P3 = nn.ParameterList()
        self.P4 = nn.ParameterList()

        self.device = device
        self.batch_norms = torch.nn.ModuleList()

        self.activation = nn.LeakyReLU()
        # self.activation = nn.Tanh()

        self.dim = [2] + list(hidden_dim) + [2]
        for i in range(len(self.dim) - 2):
            self.batch_norms.append(nn.BatchNorm2d(self.dim[i + 1]))
        for i in range(len(self.dim) - 1):
            ini = 1.0 / torch.sqrt(torch.FloatTensor([self.dim[i + 1] * self.dim[i]]))
            self.P1.append(nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
            self.P2.append(nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
            self.P3.append(nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
            self.P4.append(nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
            #self.bias.append(nn.Parameter(torch.rand([1,self.dim[i + 1],1,1], requires_grad=True) * 2 * ini - ini))

    def forward(self, H0, NRF):
        #H0:[BS,2,KNs,Nt]
        BS, _, K, Nt = H0.size()
        PEmat = simplerank(H0).unsqueeze(1)*torch.ones([1, 2, 1, 1]).to(self.device) #[BS,1,K,K],right matmul
        F = dft_matrix(NRF)[:,:,0:K].unsqueeze(0).to(self.device)#.flip(dims=[-1])
        F = complex_matmul(F,PEmat)
        WRF0 = complex_matmul(F,H0)
        A = WRF0
        A = A.contiguous() #[BS,2,NRF,Nt]

        H0_Re = H0[:,0:1,:,:]; H0_Im = H0[:,1:2,:,:] #####conjugate
        H0h = torch.concat((H0_Re, -H0_Im), dim=1).permute(0,3,2,1).unsqueeze(1)#(BS, 1, Nt, K, 2)
        H0 = torch.concat((H0_Re, H0_Im), dim=1).permute(0,2,3,1).unsqueeze(1)#(BS, 1, K, Nt, 2)
        # H0hH0 = complex_matmul(H0h,H0).permute(0,2,3,1).unsqueeze(1)/Nt/K #[BS,1,Nt,Nt,2]
        for i in range(len(self.P1)):
            A = A.view(BS, int(A.shape[1]/2), 2, NRF, Nt).permute(0,1,3,4,2)#(BS, C/2, NRF, Nt, 2)
            attention_raw = complex_matmul_model(A, H0h * torch.ones([1, int(A.shape[1]), 1, 1, 1]).to(self.device))/Nt#torch.sqrt(torch.tensor(Nt))
            #attention_raw:[BS,C/2,NRF,K,2]
            attention_norm = complex_norm2_model(attention_raw) #[BS,C/2,NRF,K]
            attention_normed = attention_raw / attention_norm.unsqueeze(-1) #[BS, C/2, NRF, K, 2]
            # 创建softmax层，指定维度
            softmax = nn.Softmax(dim=-1)
            # 应用softmax
            attention_inv = softmax(-attention_norm**4)#[BS, C/2, NRF, K]
            # attention_inv = F.softmax(-attention_norm**2, dim=-1)#[BS, C/2, NRF, K]
            attention = attention_normed * attention_inv.unsqueeze(-1) #[BS, C/2, NRF, K, 2]
            # attention = torch.concat((attention_inv.unsqueeze(-1), torch.zeros([BS, int(attention_inv.shape[1]), NRF, K, 1]).to(self.device)), dim=-1)  ### dim(Batch_size, channel/2, K, Nt, 4)
            C = complex_matmul_model(attention, H0 * torch.ones([1, int(A.shape[1]), 1, 1, 1]).to(self.device))

            A = torch.reshape(A.permute(0, 1, 4, 2, 3), [-1, int(A.shape[1] * A.shape[-1]), NRF, Nt])
            # B = torch.reshape(B.permute(0, 1, 4, 2, 3), [-1, int(B.shape[1] * B.shape[-1]), NRF, Nt])
            # C = torch.concat((A, B), dim=-1)  ### dim(Batch_size, channel/2, K, Nt, 4)
            C = torch.reshape(C.permute(0, 1, 4, 2, 3), [-1, int(C.shape[1] * C.shape[-1]), NRF, Nt])
            A1 = torch.matmul(self.P1[i], A.reshape([BS, self.dim[i], -1])).view([BS, self.dim[i + 1], NRF, Nt])
            # A2 = torch.matmul(self.P2[i], torch.mean(A, -1).view(BS, self.dim[i], -1)).view(BS,self.dim[i + 1],NRF, 1)
            # A3 = torch.matmul(self.P3[i], torch.mean(A, -2).view(BS, self.dim[i], -1)).view(BS,self.dim[i + 1], 1, Nt)
            A4 = torch.matmul(self.P4[i], C.reshape([BS, self.dim[i], -1])).view([BS, self.dim[i + 1], NRF, Nt])

            A = A1 + A4
            # 激活
            if i != len(self.P1) - 1:
                # A = self.activation(A)
                A = self.batch_norms[i](A)
                A = self.activation(A)

        # A:[BS,2,NRF,Nt]
        WRF_norm = torch.norm(A, dim=1, keepdim=True)
        WRF = A / WRF_norm / torch.sqrt(torch.tensor(Nt, dtype=torch.float32).to(self.device))
        return WRF

class EdgeGRNN_simp_softmax1(nn.Module):
    """
    2D vanilla GNN with H residual(a model induced),dont consider M^{-2}
    """
    def __init__(self, hidden_dim, device):
        super(EdgeGRNN_simp_softmax1, self).__init__()
        self.P1 = nn.ParameterList()
        self.P2 = nn.ParameterList()
        self.P3 = nn.ParameterList()
        self.P4 = nn.ParameterList()

        self.device = device
        self.batch_norms = torch.nn.ModuleList()

        self.activation = nn.LeakyReLU()

        self.dim = [2] + list(hidden_dim) + [2]
        for i in range(len(self.dim) - 2):
            self.batch_norms.append(nn.BatchNorm2d(self.dim[i + 1]))
        for i in range(len(self.dim) - 1):
            ini = 1.0 / torch.sqrt(torch.FloatTensor([self.dim[i + 1] * self.dim[i]]))
            self.P1.append(nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
            self.P2.append(nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
            self.P3.append(nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
            self.P4.append(nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
            #self.bias.append(nn.Parameter(torch.rand([1,self.dim[i + 1],1,1], requires_grad=True) * 2 * ini - ini))

    def forward(self, H0, NRF):
        #H0:[BS,2,KNs,Nt]
        BS, _, K, Nt = H0.size()
        PEmat = simplerank(H0).unsqueeze(1)*torch.ones([1, 2, 1, 1]).to(self.device) #[BS,1,K,K],right matmul
        F = dft_matrix(NRF)[:,:,0:K].unsqueeze(0).to(self.device)#.flip(dims=[-1])
        F = complex_matmul(F,PEmat)
        WRF0 = complex_matmul(F,H0)
        A = WRF0
        A = A.contiguous() #[BS,2,NRF,Nt]

        H0_Re = H0[:,0:1,:,:]; H0_Im = H0[:,1:2,:,:] #####conjugate
        H0h = torch.concat((H0_Re, -H0_Im), dim=1).permute(0,3,2,1).unsqueeze(1)#(BS, 1, Nt, K, 2)
        H0 = torch.concat((H0_Re, H0_Im), dim=1).permute(0,2,3,1).unsqueeze(1)#(BS, 1, K, Nt, 2)
        # H0hH0 = complex_matmul(H0h,H0).permute(0,2,3,1).unsqueeze(1)/Nt/K #[BS,1,Nt,Nt,2]
        for i in range(len(self.P1)):
            A = A.view(BS, int(A.shape[1]/2), 2, NRF, Nt).permute(0,1,3,4,2)#(BS, C/2, NRF, Nt, 2)
            attention_raw = complex_matmul_model(A, H0h * torch.ones([1, int(A.shape[1]), 1, 1, 1]).to(self.device))
            #attention_raw:[BS,C/2,NRF,K,2]
            attention_norm = complex_norm2_model(attention_raw) #[BS,C/2,NRF,K]
            # attention_normed = attention_raw / attention_norm.unsqueeze(-1) #[BS, C/2, NRF, K, 2]
            # 创建softmax层，指定维度
            softmax = nn.Softmax(dim=-1)
            # 应用softmax
            attention_inv = softmax(-attention_norm**4)#[BS, C/2, NRF, K]
            # attention_inv = F.softmax(-attention_norm**2, dim=-1)#[BS, C/2, NRF, K]
            # attention = attention_normed * attention_inv.unsqueeze(-1) #[BS, C/2, NRF, K, 2]
            attention = torch.concat((attention_inv.unsqueeze(-1), torch.zeros([BS, int(attention_inv.shape[1]), NRF, K, 1]).to(self.device)), dim=-1)  ### dim(Batch_size, channel/2, K, Nt, 4)
            C = complex_matmul_model(attention, H0 * torch.ones([1, int(A.shape[1]), 1, 1, 1]).to(self.device))

            A = torch.reshape(A.permute(0, 1, 4, 2, 3), [-1, int(A.shape[1] * A.shape[-1]), NRF, Nt])
            # B = torch.reshape(B.permute(0, 1, 4, 2, 3), [-1, int(B.shape[1] * B.shape[-1]), NRF, Nt])
            # C = torch.concat((A, B), dim=-1)  ### dim(Batch_size, channel/2, K, Nt, 4)
            C = torch.reshape(C.permute(0, 1, 4, 2, 3), [-1, int(C.shape[1] * C.shape[-1]), NRF, Nt])
            A1 = torch.matmul(self.P1[i], A.reshape([BS, self.dim[i], -1])).view([BS, self.dim[i + 1], NRF, Nt])
            # A2 = torch.matmul(self.P2[i], torch.mean(A, -1).view(BS, self.dim[i], -1)).view(BS,self.dim[i + 1],NRF, 1)
            # A3 = torch.matmul(self.P3[i], torch.mean(A, -2).view(BS, self.dim[i], -1)).view(BS,self.dim[i + 1], 1, Nt)
            A4 = torch.matmul(self.P4[i], C.reshape([BS, self.dim[i], -1])).view([BS, self.dim[i + 1], NRF, Nt])

            A = A1 + A4
            # 激活
            if i != len(self.P1) - 1:
                A = self.activation(A)
                A = self.batch_norms[i](A)
                # A = self.activation(A)

        # A:[BS,2,NRF,Nt]
        WRF_norm = torch.norm(A, dim=1, keepdim=True)
        WRF = A / WRF_norm / torch.sqrt(torch.tensor(Nt, dtype=torch.float32).to(self.device))
        return WRF


class MIMOHPCGNN(nn.Module):
    def __init__(self, hidden_dimAnalog,
                hidden_dimPA, hidden_dimBF,device, is_joint):
        super(MIMOHPCGNN, self).__init__()
        self.device = device
        self.Analog = EdgeGRNN_simp(hidden_dim=hidden_dimAnalog,device=device)
        self.Digital = SepGNN(is_joint=is_joint,PAhidden_dim=hidden_dimPA,BFhidden_dim=hidden_dimBF,device=device)
        # self.Digital = rzfPGNN(PAhidden_dim=hidden_dimPA, device=device)
    def ch_pre_normalize(self, H, alpha=10e-6):
        # H:[BS,2,NRBG,KNs,Nt], ratio:scaler
        H_stren = torch.norm(H, dim=[1, 4], keepdim=True)
        H_temp = H / H_stren
        coeff = torch.log(1+H_stren/alpha)
        H_input = H_temp * coeff
        return H_input

    def forward(self,H0, NRF, Ns, Pt_W, sigma_W, max_num_sub ,is_large_scale, is_opt=True,anhang=False):
        BS,_,NRBG,K,Nr,Nt = H0.size()
        H_miso = comp_svd_getEH(H0, Ns)
        if is_large_scale:
            # H_miso = H_miso / sigma_W
            H_miso = self.ch_pre_normalize(H_miso)
        #H_miso:[BS,_,NRB,K*Ns,Nt]
        WRF = self.Analog(H_miso,NRF) #chol_psuedo:[BS,2,NRF,NRF], WRF:[BS,2,NRF,Nt]
        if is_opt:
            ch_eq, chol_psuedo = cal_ch_Heq(H_miso, WRF, self.device)
            # ch_eq, chol_psuedo = cal_svd_Heq(H_miso, WRF, self.device)
        else:
            WRFcomp = complex_conjT(WRF)
            ch_eq = complex_matmul(H_miso, WRFcomp)
        ch_eq = ch_eq * torch.sqrt(torch.tensor(NRF/Nt, dtype=torch.float32).to(self.device))
        #[BS,2,NRB,K,NRF]
        WBB_hat,p = self.Digital(ch_eq, Pt_W, anhang) #WBB_hat:[BS,2,NRBG,K,NRF]
        if is_opt:
            # temp = torch.sqrt(torch.sum(ampli2(WBB_hat), dim=3)).view([BS, 1, NRBG, K, 1])
            WBB_hat = WBB_hat*p
            WBB = complex_matmul(WBB_hat, complex_conjT(chol_psuedo)) #WBB_hat:[BS,2,K*Ns,NRF]
        else:
            W_eq = complex_matmul(WBB_hat,WRF)
            temp = torch.sqrt(torch.sum(ampli2(W_eq), dim=2)).view([BS, 1, K, 1])
            WBB = WBB_hat / temp * p
        WBB = WBB.reshape(BS,2,NRBG,K,Ns,NRF) #WBB:[BS,2,NRBG,K,Ns,NRF]
        WRF = WRF.unsqueeze(2).unsqueeze(2) #WRF:[BS,2,1,1,NRF,Nt]
        return WRF, WBB#, WBB_hat, ch_eq