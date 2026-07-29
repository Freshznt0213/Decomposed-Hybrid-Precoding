import torch

from utils import *
class PAModule(nn.Module):
    def __init__(self,is_joint, hidden, device, activatefunc = nn.Softplus):
        super(PAModule, self).__init__()
        self.batch_norms = torch.nn.ModuleList()
        self.is_joint = is_joint
        self.device = device
        self.activation = activatefunc()
        if is_joint:
            self.dim = [1] + list(hidden) + [1]
            for i in range(len(self.dim) - 2):
                self.batch_norms.append(nn.BatchNorm2d(self.dim[i + 1]))
            # power allocation, update with lower triangular shademat
            self.Pd1 = nn.ParameterList()  # itself
            self.Pd2 = nn.ParameterList()  # same col
            self.Pd3 = nn.ParameterList()  # diag

            self.Pn1 = nn.ParameterList()  # itself
            self.Pn2 = nn.ParameterList()  # row diagonal neighbor
            self.Pn3 = nn.ParameterList()  # col diagonal neighbor
            self.Pn4 = nn.ParameterList()  # same col

            for i in range(len(self.dim) - 1):
                ini = 1.0 / torch.sqrt(torch.FloatTensor([self.dim[i + 1] * self.dim[i]]))
                # diagonal weight
                self.Pd1.append(
                    nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
                self.Pd2.append(
                    nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
                self.Pd3.append(
                    nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
                # self.Pd1.append(
                #     nn.init.kaiming_uniform_(nn.Parameter(torch.empty(self.dim[i + 1], self.dim[i])),
                #                              mode='fan_in', nonlinearity='leaky_relu'))
                # self.Pd2.append(
                #     nn.init.kaiming_uniform_(nn.Parameter(torch.empty(self.dim[i + 1], self.dim[i])),
                #                              mode='fan_in', nonlinearity='leaky_relu'))
                # self.Pd3.append(
                #     nn.init.kaiming_uniform_(nn.Parameter(torch.empty(self.dim[i + 1], self.dim[i])),
                #                              mode='fan_in', nonlinearity='leaky_relu'))

                # non-diagonal weight
                self.Pn1.append(
                    nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
                self.Pn2.append(
                    nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
                self.Pn3.append(
                    nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
                self.Pn4.append(
                    nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
                # self.Pn1.append(
                #     nn.init.kaiming_uniform_(nn.Parameter(torch.empty(self.dim[i + 1], self.dim[i])),
                #                              mode='fan_in', nonlinearity='leaky_relu'))
                # self.Pn2.append(
                #     nn.init.kaiming_uniform_(nn.Parameter(torch.empty(self.dim[i + 1], self.dim[i])),
                #                              mode='fan_in', nonlinearity='leaky_relu'))
                # self.Pn3.append(
                #     nn.init.kaiming_uniform_(nn.Parameter(torch.empty(self.dim[i + 1], self.dim[i])),
                #                              mode='fan_in', nonlinearity='leaky_relu'))
                # self.Pn4.append(
                #     nn.init.kaiming_uniform_(nn.Parameter(torch.empty(self.dim[i + 1], self.dim[i])),
                #                              mode='fan_in', nonlinearity='leaky_relu'))
        else:
            self.dim = [2] + list(hidden) + [1]
            for i in range(len(self.dim) - 2):
                self.batch_norms.append(nn.BatchNorm2d(self.dim[i + 1]))
            # power allocation, update with lower triangular shademat
            self.P1 = nn.ParameterList()  # itself
            self.P2 = nn.ParameterList()  # same col
            self.P3 = nn.ParameterList()  # diag

            for i in range(len(self.dim) - 1):
                ini = 1.0 / torch.sqrt(torch.FloatTensor([self.dim[i+1]*self.dim[i]]))
                self.P1.append(
                    nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
                self.P2.append(
                    nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
                self.P3.append(
                    nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
                # self.P1.append(
                #     nn.init.kaiming_uniform_(nn.Parameter(torch.empty(self.dimPA[i + 1], self.dimPA[i])),\
                #                              mode='fan_in', nonlinearity='relu'))
                # self.P2.append(
                #     nn.init.kaiming_uniform_(nn.Parameter(torch.empty(self.dimPA[i + 1], self.dimPA[i])),\
                #                              mode='fan_in', nonlinearity='relu'))
                # self.P3.append(
                #     nn.init.kaiming_uniform_(nn.Parameter(torch.empty(self.dimPA[i + 1], self.dimPA[i])),\
                #                              mode='fan_in', nonlinearity='relu'))
    def ch_pre_normalize(self, H, bias=3):
        # H:[BS,2,NRBG,K,Nt], ratio:scaler
        H_stren = torch.norm(H, dim=[1, 4], keepdim=True)
        H_temp = H / H_stren
        coeff = torch.log10(1+H_stren/bias)
        H_input = H_temp * coeff
        return H_input

    def forward(self, H, Pt_W, anhang=False, prenorm=False):
        BS,_,NRB,K,Nt = H.size()
        if prenorm:
            H = self.ch_pre_normalize(H)
        if anhang:
            Kcomp = torch.sqrt(torch.tensor(K, dtype=torch.float32).to(self.device))
            return 1 / Kcomp ########################sth wrong????????????
        else:
            if self.is_joint:
                H = H.contiguous()
                p_all = torch.tensor([]).to(self.device)
                for RB in range(NRB):
                    HRB = H[:,:,RB,:,:]
                    HRB_Re = HRB.permute(0, 3, 2, 1)[..., 0:1];  HRB_Im = -1 * HRB.permute(0, 3, 2, 1)[..., 1:2]  #####conjugate
                    HH = torch.concat((HRB_Re, HRB_Im), dim=3).permute(0, 3, 1, 2)  ####conjugate (Batch_size,channel0 = 2, Nt, K)
                    HHH = complex_matmul(HRB, HH)
                    HHHnorm = torch.norm(HHH, dim=1)  # (BS,1,K,K)
                    A = HHHnorm

                    eye_mat = torch.eye(K).unsqueeze(0).unsqueeze(0).to(self.device)
                    for i in range(len(self.Pd1)):
                        AD = A * eye_mat
                        # AT = torch.transpose(A, -1, -2).contiguous()
                        # diagonal update
                        HPd1 = torch.matmul(self.Pd1[i],A.view([BS, self.dim[i], -1]))\
                            .view([BS, self.dim[i + 1], K, K])

                        HPd2 = torch.matmul(self.Pd2[i],torch.mean(A, -2).view(BS, self.dim[i], -1)) \
                            .view(BS, self.dim[i + 1], 1, K)

                        HPd3 = torch.matmul(self.Pd3[i],torch.mean(torch.sum(AD, -1), -1).view(BS, self.dim[i], -1)) \
                            .view(BS, self.dim[i + 1], 1, 1)

                        HPd = HPd1 + 0.1 * HPd2 + 0.1 * HPd3  # applicable to diagnal element only

                        # non-diagnal update
                        HPn1 = torch.matmul(self.Pn1[i],A.view([BS, self.dim[i], -1])) \
                            .view([BS, self.dim[i + 1], K, K])

                        HPn2 = torch.matmul(self.Pn2[i],(torch.sum(AD, -2, keepdim=True) + torch.sum(AD, -1, keepdim=True)) \
                                            .view(BS, self.dim[i], -1)) \
                            .view(BS, self.dim[i + 1], K, K)

                        HPn3 = torch.matmul(self.Pn3[i],(torch.mean(A, -2, keepdim=True) + torch.mean(A, -1, keepdim=True)) \
                                            .view(BS, self.dim[i], -1)) \
                            .view(BS, self.dim[i + 1], K, K)

                        HPn4 = torch.matmul(self.Pn4[i],torch.mean(torch.sum(AD, -1), -1).view(BS, self.dim[i], -1)) \
                            .view(BS, self.dim[i + 1], 1, 1)

                        HPn = HPn1 + 0.1 * HPn2 + 0.1 * HPn3 + 0.05 * HPn4  # applicable to non-diagnal element only

                        A = HPd * eye_mat + HPn * (1 - eye_mat)

                        # activattion within the layers
                        if i != len(self.Pd1) - 1:
                            A = self.activation(A)
                            A = self.batch_norms[i](A)
                            # A = self.activation(A)
                    p1 = torch.sum(A,-1,keepdim=True)
                    p2 = torch.sum(A * eye_mat, dim=-1, keepdim=True)  # (BS,1,K,1)
                    p = p1 + p2
                    # p = nn.Sigmoid()(p)
                    # p = nn.ReLU()(p)
                    p = nn.Softplus()(p)
                    pnorm = torch.norm(p, dim=-2, keepdim=True)
                    p = torch.where(pnorm != 0, p / pnorm, p)
                    p = p * torch.sqrt(torch.tensor(Pt_W))
                    p_all = torch.cat((p_all, p.unsqueeze(2)), dim=2)
                return p_all
            else:
                H = H.contiguous()
                A = H
                epsilon = 10e-7
                for i in range(len(self.P1)):
                    # diagonal update
                    HP1 = torch.matmul(self.P1[i],A.view([BS, self.dim[i], -1])) \
                        .view([BS, self.dim[i + 1], K, Nt])

                    HP2 = torch.matmul(self.P2[i],torch.mean(A, -1).view(BS, self.dim[i], -1)) \
                        .view(BS, self.dim[i + 1], K, 1)

                    HP3 = torch.matmul(self.P3[i],torch.mean(A, -2).view(BS, self.dim[i], -1)) \
                        .view(BS, self.dim[i + 1], 1, Nt)

                    A = HP1 + 0.1*HP2 + 0.1*HP3  # applicable to diagnal element only

                    # activattion within the layers
                    if i != len(self.P1) - 1:
                        A = self.activation(A)
                        A = self.batch_norms[i](A)
                        # A = self.activation(A)
                    # A [BS,1,K,Nt]

                p = torch.sum(A, dim=-1, keepdim=True)
                # p = nn.Sigmoid()(p)
                p = nn.Softplus()(p)
                p = torch.sqrt(p / (torch.sum(p,dim=-2,keepdim=True)+epsilon))
                # pnorm = torch.norm(p, dim=-2, keepdim=True)
                # p = torch.where(pnorm != 0, p / pnorm, p)
                #
                # if torch.any(torch.isnan(p)):
                #     print(A,H)
                # if torch.all(p==0):
                #     print('frcd')
                # p = torch.sqrt(torch.softmax(p,dim=-2))
                return p
class BFModule(nn.Module):
    def __init__(self, hidden_dim, device,activatefunc=nn.Tanh): ###don't consider concatination here
        super(BFModule, self).__init__()
        self.batch_norms = torch.nn.ModuleList()
        self.device = device
        self.P1 = nn.ParameterList()#只有3个权重
        # self.P2 = nn.ParameterList()
        self.P3 = nn.ParameterList()

        self.dim = [2] + list(hidden_dim) + [2] #### input = 2, output = 2 consider concatination here(output dim is 2, just V real and imag)
        for i in range(len(self.dim)-2):
            self.batch_norms.append(nn.BatchNorm2d(self.dim[i + 1]))
        for i in range(len(self.dim) - 1):
            ini = 1.0 / torch.sqrt(torch.FloatTensor([self.dim[i + 1]*self.dim[i]]))
            self.P1.append(nn.Parameter(torch.rand([self.dim[i + 1], 2*self.dim[i]], requires_grad=True) * 2 * ini - ini))
            # self.P2.append(nn.Parameter(torch.rand([self.dim[i + 1], 2*self.dim[i]], requires_grad=True) * 2 * ini - ini))
            self.P3.append(nn.Parameter(torch.rand([self.dim[i + 1], 2*self.dim[i]], requires_grad=True) * 2 * ini - ini))

        self.activation = activatefunc()
    def ch_pre_normalize(self, H, bias=4):
        # H:[BS,2,NRBG, K,Nt], ratio:scaler
        H_stren = torch.norm(H, dim=[1, 4], keepdim=True)
        H_temp = H / H_stren
        coeff = torch.log10(H_stren) + bias
        H_input = H_temp * coeff
        return H_input

    def forward(self, H0, prenorm=False):
    #### A(Batch_x):(Batch_size, 2, NRBG, K ,Nt)
        BS,_,NRBG, K,Nt = H0.size()
        if prenorm:
            H0 = self.ch_pre_normalize(H0)

        v_all = torch.tensor([]).to(self.device)
        for RB in range(NRBG):
            HRB = H0[:,:,RB,:,:]
            HRB_Re = HRB.permute(0, 3, 2, 1)[..., 0:1]; HRB_Im = -1 * HRB.permute(0, 3, 2, 1)[..., 1:2] #####conjugate
            HRBh = torch.concat((HRB_Re, HRB_Im), dim=3)####conjugate (Batch_size, Nt, K, channel0 = 2)
            HRBh = torch.unsqueeze(HRBh, dim=1) ### (Batch_size, 1, Nt, K, channel0 = 2)
            D = HRB#
            for i in range(len(self.P1)):
                D = D.view(BS, int(D.shape[1]/2), 2, K, Nt).permute(0,1,3,4,2)
                ####H0h:(Batch_size, 1, Nt, K, 2) ; D:(Batch_size, channel/2, K, Nt, 2)
                Alpha = complex_matmul_model(D, HRBh * torch.ones([1, int(D.shape[1]), 1, 1, 1]).to(self.device))#this function is to calculate tensors with channels on the last dim
                ####(Batch_size, channel/2, K, K, 2)
                B = complex_matmul_model(Alpha,D)/Nt
                C = torch.concat((D, B), dim=-1) ### dim(Batch_size, channel/2, K, Nt, 4)
                C = torch.reshape(C.permute(0, 1, 4, 2, 3), [-1, int(C.shape[1]*C.shape[-1]), K, Nt])
                D1 = torch.matmul(self.P1[i], C.view([BS, 2*self.dim[i], -1])).view([BS, self.dim[i+1], K, Nt])
                # D2 = torch.matmul(self.P2[i],torch.mean(C,-1).view(BS,2*self.dim[i],-1)).view(BS,self.dim[i+1],K,1)
                D3 = torch.matmul(self.P3[i],torch.mean(C,-2).view(BS,2*self.dim[i],-1)).view(BS,self.dim[i+1],1,Nt)

                D = D1 + 0.1*D3
                #激活
                if i != len(self.P1)-1:
                    D = self.activation(D)
                    D = self.batch_norms[i](D)
                    # D = self.activation(D)

            temp = torch.sqrt(torch.sum(ampli2(D),dim=2)).view([BS,1,K,1])
            v = D/temp
            v_all = torch.cat((v_all, v.unsqueeze(2)), dim=2)
        return v_all

class BFModule_simp(nn.Module):
    def __init__(self, hidden_dim, device,activatefunc=nn.Tanh): ###don't consider concatination here
        super(BFModule_simp, self).__init__()
        self.batch_norms = torch.nn.ModuleList()
        self.device = device
        self.P1 = nn.ParameterList()#只有3个权重
        self.P2 = nn.ParameterList()
        self.P3 = nn.ParameterList()

        self.dim = [2] + list(hidden_dim) + [2] #### input = 2, output = 2 consider concatination here(output dim is 2, just V real and imag)
        for i in range(len(self.dim)-2):
            self.batch_norms.append(nn.BatchNorm2d(self.dim[i + 1]))
        for i in range(len(self.dim) - 1):
            ini = 1.0 / torch.sqrt(torch.FloatTensor([self.dim[i + 1]*self.dim[i]]))
            self.P1.append(nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
            self.P2.append(nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
            self.P3.append(nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))

        self.activation = activatefunc()
    def ch_pre_normalize(self, H, bias=4):
        # H:[BS,2,K,Nt], ratio:scaler
        H_stren = torch.norm(H, dim=[1, 3], keepdim=True)
        H_temp = H / H_stren
        coeff = torch.log10(H_stren) + bias
        H_input = H_temp * coeff
        return H_input

    def forward(self, H0, prenorm=False):
    #### A(Batch_x):(Batch_size, 2, K , Nt)
        BS,_,NRBG, K,Nt = H0.size()
        if prenorm:
            H0 = self.ch_pre_normalize(H0)

        v_all = torch.tensor([]).to(self.device)
        for RB in range(NRBG):
            HRB = H0[:,:,RB,:,:]
            HRB_Re = HRB.permute(0, 3, 2, 1)[..., 0:1]; HRB_Im = -1 * HRB.permute(0, 3, 2, 1)[..., 1:2] #####conjugate
            HRBh = torch.concat((HRB_Re, HRB_Im), dim=3)####conjugate (Batch_size, Nt, K, channel0 = 2)
            HRBh = torch.unsqueeze(HRBh, dim=1) ### (Batch_size, 1, Nt, K, channel0 = 2)
            D = HRB#
            for i in range(len(self.P1)):
                D = D.view(BS, int(D.shape[1]/2), 2, K, Nt).permute(0,1,3,4,2)
                ####H0h:(Batch_size, 1, Nt, K, 2) ; D:(Batch_size, channel/2, K, Nt, 2)
                Alpha = complex_matmul_model(D, HRBh * torch.ones([1, int(D.shape[1]), 1, 1, 1]).to(self.device))#this function is to calculate tensors with channels on the last dim
                ####(Batch_size, channel/2, K, K, 2)
                B = complex_matmul_model(Alpha,D)/Nt
                # C = torch.concat((D, B), dim=-1) ### dim(Batch_size, channel/2, K, Nt, 4)
                B = torch.reshape(B.permute(0, 1, 4, 2, 3), [-1, int(B.shape[1] * B.shape[-1]), K, Nt])
                D = torch.reshape(D.permute(0, 1, 4, 2, 3), [-1, int(D.shape[1] * D.shape[-1]), K, Nt])
                # C = torch.reshape(C.permute(0, 1, 4, 2, 3), [-1, int(C.shape[1]*C.shape[-1]), K, Nt])
                D1 = torch.matmul(self.P1[i], D.view([BS, self.dim[i], -1])).view([BS, self.dim[i+1], K, Nt])
                # D2 = torch.matmul(self.P2[i],torch.mean(C,-1).view(BS,2*self.dim[i],-1)).view(BS,self.dim[i+1],K,1)
                # D3 = torch.matmul(self.P3[i],torch.mean(C,-2).view(BS,2*self.dim[i],-1)).view(BS,self.dim[i+1],1,Nt)
                D3 = torch.matmul(self.P3[i], B.view([BS, self.dim[i], -1])).view([BS, self.dim[i + 1], K, Nt])
                D = D1 + 0.1*D3 #0.1*D2 + 0.1*D3
                #激活
                if i != len(self.P1)-1:
                    D = self.activation(D)
                    D = self.batch_norms[i](D)
                    # D = self.activation(D)

            temp = torch.sqrt(torch.sum(ampli2(D),dim=2)).view([BS,1,K,1])
            v = D/temp
            v_all = torch.cat((v_all, v.unsqueeze(2)), dim=2)
        return v_all

class SepGNN(nn.Module):
    def __init__(self, is_joint, PAhidden_dim, BFhidden_dim, device): ###don't consider concatination here
        super(SepGNN, self).__init__()
        self.PAmodule = PAModule(is_joint=is_joint, hidden=PAhidden_dim, device=device)
        self.BFmodule = BFModule_simp(hidden_dim=BFhidden_dim,device=device)

    def forward(self, H0, Pt_W,anhang=False):
        V = self.BFmodule(H0)
        p = self.PAmodule(H0, Pt_W, anhang=anhang)
        return V, p