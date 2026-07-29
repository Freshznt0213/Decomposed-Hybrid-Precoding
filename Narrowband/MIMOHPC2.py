import torch
import torch.nn as nn
from utils import *
from SepGNN import SepGNN
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
    # H: 输入复数张量，形状为 [Batchsize, 2, K, Nt]
    # 返回：形状为 [Batchsize, K, K] 的置换矩阵

    H = torch.complex(H[:,0,:,:],H[:,1,:,:])
    #[BS,K,Nt]
    # Step 1: 计算每个样本的行向量 L2 范数
    norms = torch.norm(H, dim=2)  # [Batchsize, K]，每行向量的 L2 范数

    # Step 2: 对 norms 排序，得到排序后的索引
    _, sorted_indices = torch.sort(norms, dim=1)  # [Batchsize, K]，按 norm 从小到大的索引

    # Step 3: 生成置换矩阵
    # 创建一个初始矩阵，形状为 [Batchsize, K, K]，用于存储置换矩阵
    PEmat = torch.zeros(H.size(0), H.size(1), H.size(1), dtype=torch.float32, device=H.device)

    # 使用 gather 和 one-hot 编码生成置换矩阵
    PEmat.scatter_(2, sorted_indices.unsqueeze(2), 1.0)
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
    BS, _, K, Nt = H0.size()
    NRF = WRF.size(2)
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
    #H0:[BS,2,K,Nt], WRF:[BS,2,NRF,Nt]
    BS, _, K, Nt = H0.size()
    NRF = WRF.size(2)
    WRF_comp = torch.complex(WRF[:,0],-WRF[:,1]).transpose(-1,-2) #[BS,Nt,NRF]
    WRF_U,WRF_D,WRF_Vh = torch.linalg.svd(WRF_comp) # U:[BS,Nt,Nt],sigma:[BS,NRF],Vh:[BS,NRF,NRF]
    WRF_D_inv = torch.diag_embed(1 / WRF_D).to(torch.complex64)
    semi_unitary_mat = torch.matmul(WRF_U[:,:,0:NRF],WRF_Vh)
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
            ini = 1.0 / torch.sqrt(torch.FloatTensor([self.dim[i + 1] * self.dim[i] * 10]))
            self.P1.append(nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
            # self.P2.append(nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
            # self.P3.append(nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
            self.P4.append(nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
            #self.bias.append(nn.Parameter(torch.rand([1,self.dim[i + 1],1,1], requires_grad=True) * 2 * ini - ini))

    def forward(self, H0, NRF):
        #H0:[BS,2,KNs,Nt]
        BS, _, K, Nt = H0.size()
        PEmat = simplerank(H0).unsqueeze(1)*torch.ones([1, 2, 1, 1]).to(self.device) #[BS,1,K,K],right matmul
        # F = dft_matrix(NRF)[:,:,0:K].unsqueeze(0).to(self.device)#.flip(dims=[-1])
        F = random_unitary_matrix(NRF)[:,:,0:K].unsqueeze(0).to(self.device)
        F = complex_matmul(F,PEmat)
        WRF0 = complex_matmul(F,H0)
        A = WRF0
        # A = H0
        A = A.contiguous() #[BS,2,NRF,Nt]

        H0_Re = H0.permute(0, 1, 3, 2)[:,0:1,:,:]; H0_Im = -1 * H0.permute(0, 1, 3, 2)[:,1:2,:,:] #####conjugate
        H0h = torch.concat((H0_Re, H0_Im), dim=1)####conjugate (Batch_size, 2, Nt, K)
        H0hH0 = complex_matmul(H0h,H0).permute(0,2,3,1).unsqueeze(1)/Nt/K #[BS,1,Nt,Nt,2]
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

# class EdgeGRNN_simp(nn.Module):
#     """
#     2D vanilla GNN with H residual(a model induced),dont consider M^{-2}.
#     Refactored to use nn.Linear for FLOPs calculation.
#     """
#
#     def __init__(self, hidden_dim, device):
#         super(EdgeGRNN_simp, self).__init__()
#
#         self.device = device
#         self.dim = [2] + list(hidden_dim) + [2]
#
#         self.linear1 = nn.ModuleList()
#         # self.P2 = nn.ParameterList()
#         # self.P3 = nn.ParameterList()
#         self.linear4 = nn.ModuleList()
#
#         self.batch_norms = torch.nn.ModuleList()
#         self.activation = nn.LeakyReLU()
#
#         for i in range(len(self.dim) - 2):
#             self.batch_norms.append(nn.BatchNorm2d(self.dim[i + 1]))
#
#         for i in range(len(self.dim) - 1):
#             # Create Linear layers
#             layer1 = nn.Linear(self.dim[i], self.dim[i + 1], bias=False)
#             layer4 = nn.Linear(self.dim[i], self.dim[i + 1], bias=False)
#
#             # Custom Initialization
#             ini = 1.0 / torch.sqrt(torch.FloatTensor([self.dim[i + 1] * self.dim[i] * 10]))
#             layer1.weight.data.uniform_(-ini.item(), ini.item())
#             layer4.weight.data.uniform_(-ini.item(), ini.item())
#
#             self.linear1.append(layer1)
#             self.linear4.append(layer4)
#
#             # Original ParameterList code for reference
#             # self.P1.append(nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
#             # self.P4.append(nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
#
#     def forward(self, H0, NRF):
#         # H0:[BS,2,KNs,Nt]
#         BS, _, K, Nt = H0.size()
#
#         PEmat = simplerank(H0).unsqueeze(1) * torch.ones([1, 2, 1, 1]).to(self.device)  # \[BS,1,K,K\],right matmul
#         F = dft_matrix(NRF)[:, :, 0:K].unsqueeze(0).to(self.device)  # .flip(dims=\[-1\])
#         # F = random_unitary_matrix(NRF)[:, :, 0:K].unsqueeze(0).to(self.device)
#         F = complex_matmul(F, PEmat)
#         WRF0 = complex_matmul(F, H0)
#         A = WRF0
#         A = A.contiguous()  # \[BS,2,NRF,Nt\]
#
#         H0_Re = H0.permute(0, 1, 3, 2)[:, 0:1, :, :];
#         H0_Im = -1 * H0.permute(0, 1, 3, 2)[:, 1:2, :, :]  #####conjugate
#         H0h = torch.concat((H0_Re, H0_Im), dim=1)  ####conjugate (Batch_size, 2, Nt, K)
#         H0hH0 = complex_matmul(H0h, H0).permute(0, 2, 3, 1).unsqueeze(1) / Nt / K  # \[BS,1,Nt,Nt,2\]
#
#         for i in range(len(self.linear1)):
#             c_in = self.dim[i]
#             c_out = self.dim[i + 1]
#
#             A = A.view(BS, int(A.shape[1] / 2), 2, NRF, Nt).permute(0, 1, 3, 4, 2)
#             ####H0h:(Batch_size, 1, Nt, K, 2) ; D:(Batch_size, channel/2, K, Nt, 2)
#             B = complex_matmul_model(A, H0hH0 * torch.ones([1, int(A.shape[1]), 1, 1, 1]).to(self.device))
#
#             A = torch.reshape(A.permute(0, 1, 4, 2, 3), [-1, int(A.shape[1] * A.shape[-1]), NRF, Nt])
#             B = torch.reshape(B.permute(0, 1, 4, 2, 3), [-1, int(B.shape[1] * B.shape[-1]), NRF, Nt])
#
#             # --- Replacing matmul with nn.Linear ---
#             # To use nn.Linear, the feature dimension (c_in) must be the last one.
#             # Original shape: [BS, c_in, NRF, Nt]
#             # We reshape to [BS, c_in, L] where L=NRF*Nt, then permute to [BS, L, c_in]
#
#             A_in = A.view([BS, c_in, -1]).permute(0, 2, 1)  # Shape: [BS, NRF*Nt, c_in]
#             B_in = B.view([BS, c_in, -1]).permute(0, 2, 1)  # Shape: [BS, NRF*Nt, c_in]
#
#             # Apply linear layers
#             A1_out = self.linear1[i](A_in)  # Shape: [BS, NRF*Nt, c_out]
#             A4_out = self.linear4[i](B_in)  # Shape: [BS, NRF*Nt, c_out]
#
#             # Permute back to [BS, c_out, NRF*Nt] and reshape
#             A1 = A1_out.permute(0, 2, 1).view([BS, c_out, NRF, Nt])
#             A4 = A4_out.permute(0, 2, 1).view([BS, c_out, NRF, Nt])
#
#             # --- End of replacement ---
#
#             A = A1 + A4
#
#             # 激活
#             if i != len(self.linear1) - 1:
#                 A = self.batch_norms[i](A)
#                 A = self.activation(A)
#
#         # A:[BS,2,NRF,Nt]
#         WRF_norm = torch.norm(A, dim=1, keepdim=True)
#         WRF = A / WRF_norm / torch.sqrt(torch.tensor(Nt, dtype=torch.float32).to(self.device))
#
#         return WRF


class MIMOHPCGNN(nn.Module):
    def __init__(self, hidden_dimAnalog,
                 is_joint, hidden_dimPA, hidden_dimBF,device):
        super(MIMOHPCGNN, self).__init__()
        self.device = device
        self.Analog = EdgeGRNN_simp(hidden_dim=hidden_dimAnalog,device=device)
        self.Digital = SepGNN(is_joint=is_joint,PAhidden_dim=hidden_dimPA,BFhidden_dim=hidden_dimBF,device=device)

    def forward(self,H0, NRF, Ns, is_opt=True,anhang=False):
        if H0.dim()==4:
            H0 = H0.unsqueeze(-2)
            H_miso = H0
        else:
            H_miso = comp_svd_getEH(H0, Ns)
        H_miso = H_miso.contiguous()
        BS,_,K,Nr,Nt = H0.size()
        #H_miso:[BS,_,K,Ns,Nt]
        H_miso = H_miso.reshape(BS,2,K*Ns,Nt)
        # H_miso = H_miso*torch.sqrt(torch.tensor(self.Ns/Nr))
        WRF = self.Analog(H_miso,NRF) #chol_psuedo:[BS,2,NRF,NRF], WRF:[BS,2,NRF,Nt]
        if is_opt:
            ch_eq, chol_psuedo = cal_ch_Heq(H_miso, WRF, self.device)
            # ch_eq, chol_psuedo = cal_svd_Heq(H_miso, WRF, self.device)
        else:
            WRFcomp = complex_conjT(WRF)
            ch_eq = complex_matmul(H_miso, WRFcomp)
        ch_eq = ch_eq * torch.sqrt(torch.tensor(NRF/Nt, dtype=torch.float32).to(self.device))
        ch_eq = ch_eq.contiguous()
        WBB_hat,p = self.Digital(ch_eq, Ns,anhang) #WBB_hat:[BS,2,K*Ns,NRF]
        if is_opt:
            temp = torch.sqrt(torch.sum(ampli2(WBB_hat), dim=2)).view([BS, 1, K*Ns, 1])
            WBB_hat = WBB_hat/temp*p
            WBB = complex_matmul(WBB_hat, complex_conjT(chol_psuedo)) #WBB_hat:[BS,2,K*Ns,NRF]
        else:
            W_eq = complex_matmul(WBB_hat,WRF)
            temp = torch.sqrt(torch.sum(ampli2(W_eq), dim=2)).view([BS, 1, K*Ns, 1])
            WBB = WBB_hat / temp * p
        WBB = WBB.reshape(BS,2,K,Ns,NRF)
        WRF = WRF.unsqueeze(2)
        return WRF, WBB#, WBB_hat, ch_eq