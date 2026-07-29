from utils import *
from torch.cuda import Stream
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
        # H:[BS,2,K,Nt], ratio:scaler
        H_stren = torch.norm(H, dim=[1, 3], keepdim=True)
        H_temp = H / H_stren
        coeff = torch.log10(1+H_stren/bias)
        H_input = H_temp * coeff
        return H_input

    def forward(self, H, anhang=False, prenorm=True):
        BS,_,K,Nt = H.size()
        if prenorm:
            H = self.ch_pre_normalize(H)
        if anhang:
            Kcomp = torch.sqrt(torch.tensor(K, dtype=torch.float32).to(self.device))
            return 1 / Kcomp
        else:
            if self.is_joint:
                H = H.contiguous()
                H_Re = H.permute(0, 3, 2, 1)[..., 0:1];  H_Im = -1 * H.permute(0, 3, 2, 1)[..., 1:2]  #####conjugate
                HH = torch.concat((H_Re, H_Im), dim=3).permute(0, 3, 1, 2)  ####conjugate (Batch_size,channel0 = 2, Nt, K)
                HHH = complex_matmul(H, HH)
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
                return p
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



# Note: The 'complex_matmul' function is assumed to be defined elsewhere in your code.

# class PAModule(nn.Module):
#     """
#     A refactored version of the PAModule where torch.matmul operations
#     involving learnable parameters are replaced with nn.Linear layers for FLOPs
#     calculation compatibility.
#     """
#
#     def __init__(self, is_joint, hidden, device, activatefunc=nn.Softplus):
#         super(PAModule, self).__init__()
#         self.batch_norms = torch.nn.ModuleList()
#         self.is_joint = is_joint
#         self.device = device
#         self.activation = activatefunc()
#
#         if is_joint:
#             self.dim = [1] + list(hidden) + [1]
#             for i in range(len(self.dim) - 2):
#                 self.batch_norms.append(nn.BatchNorm2d(self.dim[i + 1]))
#
#             # ModuleLists for Linear layers
#             self.linear_Pd1, self.linear_Pd2, self.linear_Pd3 = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
#             self.linear_Pn1, self.linear_Pn2, self.linear_Pn3, self.linear_Pn4 = nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
#
#             param_lists = [self.linear_Pd1, self.linear_Pd2, self.linear_Pd3, self.linear_Pn1, self.linear_Pn2,
#                            self.linear_Pn3, self.linear_Pn4]
#
#             for i in range(len(self.dim) - 1):
#                 ini = 1.0 / torch.sqrt(torch.FloatTensor([self.dim[i + 1] * self.dim[i]]))
#                 for param_list in param_lists:
#                     layer = nn.Linear(self.dim[i], self.dim[i + 1], bias=False)
#                     layer.weight.data.uniform_(-ini.item(), ini.item())
#                     param_list.append(layer)
#         else:
#             self.dim = [2] + list(hidden) + [1]
#             for i in range(len(self.dim) - 2):
#                 self.batch_norms.append(nn.BatchNorm2d(self.dim[i + 1]))
#
#             # ModuleLists for Linear layers
#             self.linear_P1, self.linear_P2, self.linear_P3 = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
#
#             for i in range(len(self.dim) - 1):
#                 ini = 1.0 / torch.sqrt(torch.FloatTensor([self.dim[i + 1] * self.dim[i]]))
#                 for param_list in [self.linear_P1, self.linear_P2, self.linear_P3]:
#                     layer = nn.Linear(self.dim[i], self.dim[i + 1], bias=False)
#                     layer.weight.data.uniform_(-ini.item(), ini.item())
#                     param_list.append(layer)
#
#     def ch_pre_normalize(self, H, bias=3):
#         H_stren = torch.norm(H, dim=[1, 3], keepdim=True)
#         H_temp = H / H_stren
#         coeff = torch.log10(1 + H_stren / bias)
#         H_input = H_temp * coeff
#         return H_input
#
#     def _apply_linear(self, layer, x, c_in, c_out, out_shape):
#         """Helper function to apply linear layer with permutes and reshapes."""
#         # x is expected to be [BS, c_in, ...]
#         bs = x.shape[0]
#         x_reshaped = x.view(bs, c_in, -1)
#         x_permuted = x_reshaped.permute(0, 2, 1)  # [BS, L, c_in]
#
#         output = layer(x_permuted)  # [BS, L, c_out]
#
#         output_permuted = output.permute(0, 2, 1)  # [BS, c_out, L]
#         return output_permuted.view(out_shape)
#
#     def forward(self, H, anhang=False, prenorm=True):
#         BS, _, K, Nt = H.size()
#         if prenorm:
#             H = self.ch_pre_normalize(H)
#         if anhang:
#             Kcomp = torch.sqrt(torch.tensor(K, dtype=torch.float32).to(self.device))
#             return 1 / Kcomp
#
#         if self.is_joint:
#             H = H.contiguous()
#             H_Re = H.permute(0, 3, 2, 1)[..., 0:1];
#             H_Im = -1 * H.permute(0, 3, 2, 1)[..., 1:2]
#             HH = torch.concat((H_Re, H_Im), dim=3).permute(0, 3, 1, 2)
#             HHH = complex_matmul(H, HH)
#             HHHnorm = torch.norm(HHH, dim=1)
#             A = HHHnorm
#
#             eye_mat = torch.eye(K).unsqueeze(0).unsqueeze(0).to(self.device)
#             for i in range(len(self.linear_Pd1)):
#                 c_in, c_out = self.dim[i], self.dim[i + 1]
#                 AD = A * eye_mat
#
#                 # Diagonal update
#                 HPd1 = self._apply_linear(self.linear_Pd1[i], A, c_in, c_out, (BS, c_out, K, K))
#                 HPd2_in = torch.mean(A, -2)  # [BS, c_in, K]
#                 HPd2 = self._apply_linear(self.linear_Pd2[i], HPd2_in, c_in, c_out, (BS, c_out, 1, K))
#                 HPd3_in = torch.mean(torch.sum(AD, -1), -1)  # [BS, c_in]
#                 HPd3 = self._apply_linear(self.linear_Pd3[i], HPd3_in, c_in, c_out, (BS, c_out, 1, 1))
#                 HPd = HPd1 + 0.1 * HPd2 + 0.1 * HPd3
#
#                 # Non-diagonal update
#                 HPn1 = self._apply_linear(self.linear_Pn1[i], A, c_in, c_out, (BS, c_out, K, K))
#                 HPn2_in = torch.sum(AD, -2, keepdim=True) + torch.sum(AD, -1, keepdim=True)
#                 HPn2 = self._apply_linear(self.linear_Pn2[i], HPn2_in, c_in, c_out, (BS, c_out, K, K))
#                 HPn3_in = torch.mean(A, -2, keepdim=True) + torch.mean(A, -1, keepdim=True)
#                 HPn3 = self._apply_linear(self.linear_Pn3[i], HPn3_in, c_in, c_out, (BS, c_out, K, K))
#                 HPn4_in = torch.mean(torch.sum(AD, -1), -1)
#                 HPn4 = self._apply_linear(self.linear_Pn4[i], HPn4_in, c_in, c_out, (BS, c_out, 1, 1))
#                 HPn = HPn1 + 0.1 * HPn2 + 0.1 * HPn3 + 0.05 * HPn4
#
#                 A = HPd * eye_mat + HPn * (1 - eye_mat)
#
#                 if i != len(self.linear_Pd1) - 1:
#                     A = self.activation(A)
#                     A = self.batch_norms[i](A)
#
#             p1 = torch.sum(A, -1, keepdim=True)
#             p2 = torch.sum(A * eye_mat, dim=-1, keepdim=True)
#             p = p1 + p2
#             p = nn.Softplus()(p)
#             pnorm = torch.norm(p, dim=-2, keepdim=True)
#             p = torch.where(pnorm != 0, p / pnorm, p)
#             return p
#         else:  # Not is_joint
#             A = H.contiguous()
#             epsilon = 1e-7
#             for i in range(len(self.linear_P1)):
#                 c_in, c_out = self.dim[i], self.dim[i + 1]
#
#                 # Apply linear layers
#                 HP1 = self._apply_linear(self.linear_P1[i], A, c_in, c_out, (BS, c_out, K, Nt))
#                 HP2_in = torch.mean(A, -1)  # [BS, c_in, K]
#                 HP2 = self._apply_linear(self.linear_P2[i], HP2_in, c_in, c_out, (BS, c_out, K, 1))
#                 HP3_in = torch.mean(A, -2)  # [BS, c_in, Nt]
#                 HP3 = self._apply_linear(self.linear_P3[i], HP3_in, c_in, c_out, (BS, c_out, 1, Nt))
#
#                 A = HP1 + 0.1 * HP2 + 0.1 * HP3
#
#                 if i != len(self.linear_P1) - 1:
#                     A = self.activation(A)
#                     A = self.batch_norms[i](A)
#
#             p = torch.sum(A, dim=-1, keepdim=True)
#             p = nn.Softplus()(p)
#             p = torch.sqrt(p / (torch.sum(p, dim=-2, keepdim=True) + epsilon))
#             return p


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
            ini = 1.0 / torch.sqrt(torch.FloatTensor([self.dim[i + 1]*self.dim[i]*10]))
            self.P1.append(nn.Parameter(torch.rand([self.dim[i + 1], 2*self.dim[i]], requires_grad=True) * 2 * ini - ini))
            # self.P2.append(nn.Parameter(torch.rand([self.dim[i + 1], 2*self.dim[i]], requires_grad=True) * 2 * ini - ini))
            self.P3.append(nn.Parameter(torch.rand([self.dim[i + 1], 2*self.dim[i]], requires_grad=True) * 2 * ini - ini))

        self.activation = activatefunc()
    def ch_pre_normalize(self, H, bias=4):
        # H:[BS,2,K,Nt], ratio:scaler
        H_stren = torch.norm(H, dim=[1, 3], keepdim=True)
        H_temp = H / H_stren
        coeff = torch.log10(H_stren) + bias
        H_input = H_temp * coeff
        return H_input

    def forward(self, H0, Ns, prenorm=False):
    #### A(Batch_x):(Batch_size, 2, K , Nt)
        BS,_,K,Nt = H0.size()
        if prenorm:
            H0 = self.ch_pre_normalize(H0)
        H0_Re = H0.permute(0, 3, 2, 1)[..., 0:1]; H0_Im = -1 * H0.permute(0, 3, 2, 1)[..., 1:2] #####conjugate
        H0h = torch.concat((H0_Re, H0_Im), dim=3)####conjugate (Batch_size, Nt, K, channel0 = 2)
        H0h = torch.unsqueeze(H0h, dim=1) ### (Batch_size, 1, Nt, K, channel0 = 2)
        D = H0#
        for i in range(len(self.P1)):
            D = D.view(BS, int(D.shape[1]/2), 2, K, Nt).permute(0,1,3,4,2)
            ####H0h:(Batch_size, 1, Nt, K, 2) ; D:(Batch_size, channel/2, K, Nt, 2)
            Alpha = complex_matmul_model(D, H0h * torch.ones([1, int(D.shape[1]), 1, 1, 1]).to(self.device))#this function is to calculate tensors with channels on the last dim
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
        return v

class BFModule_simp(nn.Module):
    def __init__(self, hidden_dim, device,activatefunc=nn.Tanh): ###don't consider concatination here
        super(BFModule_simp, self).__init__()
        self.batch_norms = torch.nn.ModuleList()
        self.device = device
        self.P1 = nn.ParameterList()#只有3个权重
        # self.P2 = nn.ParameterList()
        # self.P3 = nn.ParameterList()
        self.P4 = nn.ParameterList()

        self.dim = [2] + list(hidden_dim) + [2] #### input = 2, output = 2 consider concatination here(output dim is 2, just V real and imag)
        for i in range(len(self.dim)-2):
            self.batch_norms.append(nn.BatchNorm2d(self.dim[i + 1]))
        for i in range(len(self.dim) - 1):
            ini = 1.0 / torch.sqrt(torch.FloatTensor([self.dim[i + 1]*self.dim[i]*10]))
            self.P1.append(nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
            # self.P2.append(nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
            # self.P3.append(nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
            self.P4.append(nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))

        self.activation = activatefunc()
    def ch_pre_normalize(self, H, bias=4):
        # H:[BS,2,K,Nt], ratio:scaler
        H_stren = torch.norm(H, dim=[1, 3], keepdim=True)
        H_temp = H / H_stren
        coeff = torch.log10(H_stren) + bias
        H_input = H_temp * coeff
        return H_input

    def forward(self, H0, Ns, prenorm=False):
    #### A(Batch_x):(Batch_size, 2, K , Nt)
        BS,_,K,Nt = H0.size()
        if prenorm:
            H0 = self.ch_pre_normalize(H0)
        H0_Re = H0.permute(0, 3, 2, 1)[..., 0:1]; H0_Im = -1 * H0.permute(0, 3, 2, 1)[..., 1:2] #####conjugate
        H0h = torch.concat((H0_Re, H0_Im), dim=3)####conjugate (Batch_size, Nt, K, channel0 = 2)
        H0h = torch.unsqueeze(H0h, dim=1) ### (Batch_size, 1, Nt, K, channel0 = 2)
        D = H0#
        for i in range(len(self.P1)):
            D = D.view(BS, int(D.shape[1]/2), 2, K, Nt).permute(0,1,3,4,2)
            ####H0h:(Batch_size, 1, Nt, K, 2) ; D:(Batch_size, channel/2, K, Nt, 2)
            Alpha = complex_matmul_model(D, H0h * torch.ones([1, int(D.shape[1]), 1, 1, 1]).to(self.device))#this function is to calculate tensors with channels on the last dim
            ####(Batch_size, channel/2, K, K, 2)
            B = complex_matmul_model(Alpha,D)/Nt
            # C = torch.concat((D, B), dim=-1) ### dim(Batch_size, channel/2, K, Nt, 4)
            B = torch.reshape(B.permute(0, 1, 4, 2, 3), [-1, int(B.shape[1] * B.shape[-1]), K, Nt])
            D = torch.reshape(D.permute(0, 1, 4, 2, 3), [-1, int(D.shape[1] * D.shape[-1]), K, Nt])
            # C = torch.reshape(C.permute(0, 1, 4, 2, 3), [-1, int(C.shape[1]*C.shape[-1]), K, Nt])
            D1 = torch.matmul(self.P1[i], D.view([BS, self.dim[i], -1])).view([BS, self.dim[i+1], K, Nt])
            # D2 = torch.matmul(self.P2[i],torch.mean(C,-1).view(BS,2*self.dim[i],-1)).view(BS,self.dim[i+1],K,1)
            # D3 = torch.matmul(self.P3[i],torch.mean(C,-2).view(BS,2*self.dim[i],-1)).view(BS,self.dim[i+1],1,Nt)
            D4 = torch.matmul(self.P4[i], B.view([BS, self.dim[i], -1])).view([BS, self.dim[i + 1], K, Nt])
            D = D1 + D4 #0.1*D2 + 0.1*D3
            #激活
            if i != len(self.P1)-1:
                D = self.activation(D)
                D = self.batch_norms[i](D)
                # D = self.activation(D)

        temp = torch.sqrt(torch.sum(ampli2(D),dim=2)).view([BS,1,K,1])
        v = D/temp
        return v


# class BFModule_simp(nn.Module):
#     """
#     A refactored version of the BFModule_simp module where torch.matmul operations
#     involving learnable parameters are replaced with nn.Linear layers for FLOPs
#     calculation compatibility.
#     """
#
#     def __init__(self, hidden_dim, device, activatefunc=nn.Tanh):
#         super(BFModule_simp, self).__init__()
#         self.batch_norms = torch.nn.ModuleList()
#         self.device = device
#
#         self.dim = [2] + list(hidden_dim) + [2]
#
#         # Define ModuleLists for Linear layers
#         self.linear1 = nn.ModuleList()
#         self.linear4 = nn.ModuleList()
#
#         for i in range(len(self.dim) - 2):
#             self.batch_norms.append(nn.BatchNorm2d(self.dim[i + 1]))
#
#         for i in range(len(self.dim) - 1):
#             # Create Linear layers
#             layer1 = nn.Linear(self.dim[i], self.dim[i + 1], bias=False)
#             layer4 = nn.Linear(self.dim[i], self.dim[i + 1], bias=False)
#
#             # Custom Initialization for the weights of Linear layers
#             ini = 1.0 / torch.sqrt(torch.FloatTensor([self.dim[i + 1] * self.dim[i] * 10]))
#             layer1.weight.data.uniform_(-ini.item(), ini.item())
#             layer4.weight.data.uniform_(-ini.item(), ini.item())
#
#             self.linear1.append(layer1)
#             self.linear4.append(layer4)
#
#         self.activation = activatefunc()
#
#     def ch_pre_normalize(self, H, bias=4):
#         # H:[BS,2,K,Nt], ratio:scaler
#         H_stren = torch.norm(H, dim=[1, 3], keepdim=True)
#         H_temp = H / H_stren
#         coeff = torch.log10(H_stren) + bias
#         H_input = H_temp * coeff
#         return H_input
#
#     def forward(self, H0, Ns, prenorm=False):
#         # A(Batch_x):(Batch_size, 2, K , Nt)
#         BS, _, K, Nt = H0.size()
#         if prenorm:
#             H0 = self.ch_pre_normalize(H0)
#
#         H0_Re = H0.permute(0, 3, 2, 1)[..., 0:1]
#         H0_Im = -1 * H0.permute(0, 3, 2, 1)[..., 1:2]
#         H0h = torch.concat((H0_Re, H0_Im), dim=3)  # conjugate (Batch_size, Nt, K, channel0 = 2)
#         H0h = torch.unsqueeze(H0h, dim=1)  # (Batch_size, 1, Nt, K, channel0 = 2)
#         D = H0
#
#         for i in range(len(self.linear1)):
#             c_in = self.dim[i]
#             c_out = self.dim[i + 1]
#
#             D = D.view(BS, int(D.shape[1] / 2), 2, K, Nt).permute(0, 1, 3, 4, 2)
#             # H0h:(Batch_size, 1, Nt, K, 2) ; D:(Batch_size, channel/2, K, Nt, 2)
#             Alpha = complex_matmul_model(D, H0h * torch.ones([1, int(D.shape[1]), 1, 1, 1]).to(self.device))
#             # (Batch_size, channel/2, K, K, 2)
#             B = complex_matmul_model(Alpha, D) / Nt
#
#             B = torch.reshape(B.permute(0, 1, 4, 2, 3), [-1, int(B.shape[1] * B.shape[-1]), K, Nt])
#             D = torch.reshape(D.permute(0, 1, 4, 2, 3), [-1, int(D.shape[1] * D.shape[-1]), K, Nt])
#
#             # --- Replacing matmul with nn.Linear ---
#             # To use nn.Linear, the feature dimension (c_in) must be the last one.
#             # We reshape from [BS, c_in, K, Nt] to [BS, K*Nt, c_in]
#             D_in = D.view([BS, c_in, -1]).permute(0, 2, 1)  # Shape: [BS, K*Nt, c_in]
#             B_in = B.view([BS, c_in, -1]).permute(0, 2, 1)  # Shape: [BS, K*Nt, c_in]
#
#             # Apply linear layers
#             D1_out = self.linear1[i](D_in)  # Shape: [BS, K*Nt, c_out]
#             D4_out = self.linear4[i](B_in)  # Shape: [BS, K*Nt, c_out]
#
#             # Permute back to [BS, c_out, K*Nt] and reshape to [BS, c_out, K, Nt]
#             D1 = D1_out.permute(0, 2, 1).view([BS, c_out, K, Nt])
#             D4 = D4_out.permute(0, 2, 1).view([BS, c_out, K, Nt])
#             # --- End of replacement ---
#
#             D = D1 + D4
#
#             # Activation
#             if i != len(self.linear1) - 1:
#                 D = self.activation(D)
#                 D = self.batch_norms[i](D)
#
#         temp = torch.sqrt(torch.sum(ampli2(D), dim=2)).view([BS, 1, K, 1])
#         v = D / temp
#         return v


class SepGNN(nn.Module):
    def __init__(self, is_joint, PAhidden_dim, BFhidden_dim, device): ###don't consider concatination here
        super(SepGNN, self).__init__()
        self.PAmodule = PAModule(is_joint=is_joint, hidden=PAhidden_dim, device=device)
        self.BFmodule = BFModule_simp(hidden_dim=BFhidden_dim,device=device)

    def forward(self, H0, Ns, anhang=False):
        p = self.PAmodule(H0,anhang=anhang)
        V = self.BFmodule(H0,Ns)
        return V, p


class SepGNNParallel(nn.Module):
    def __init__(self, is_joint, PAhidden_dim, BFhidden_dim, device):
        super(SepGNNParallel, self).__init__()
        self.PAmodule = PAModule(is_joint=is_joint, hidden=PAhidden_dim, device=device)
        self.BFmodule = BFModule(hidden_dim=BFhidden_dim, device=device)
        self.device = device

        # 创建 CUDA 流
        self.stream_pa = Stream()
        self.stream_bf = Stream()

    def forward(self, H0, Ns, anhang=False):
        # 用于存储结果
        p_result = []
        V_result = []

        # 在独立的流中并行执行
        with torch.cuda.stream(self.stream_pa):
            p = self.PAmodule(H0, anhang=anhang)
            p_result.append(p)

        with torch.cuda.stream(self.stream_bf):
            V = self.BFmodule(H0, Ns)
            V_result.append(V)

        # 同步所有流
        torch.cuda.synchronize()

        # 获取结果
        p = p_result[0]
        V = V_result[0]

        return V, p

    def forward_sequential(self, H0, Ns, anhang=False):
        """串行版本用于对比"""
        p = self.PAmodule(H0, anhang=anhang)
        V = self.BFmodule(H0, Ns)
        return V, p


class DiamondGNN(nn.Module):
    #          --->PA---
    # EHSVD ---         --->output
    #          --->BF---
    def __init__(self, is_joint, hidden_dimPA, hidden_dimBF, Ns, device):
        super(DiamondGNN, self).__init__()
        self.Ns = Ns
        self.device = device
        # self.Digital = SepGNN(is_joint=is_joint, PAhidden_dim=hidden_dimPA, BFhidden_dim=hidden_dimBF, device=device)
        self.Digital = SepTGNN(Ns=Ns,hidden_dim=hidden_dimBF,device=device)
    def forward(self,H,anhang=False,prenorm=False):
        if H.dim()==4:
            H = H.unsqueeze(-2)
        BS, _, K, Nr, Nt = H.size()
        # time_start = time.time()
        EH = comp_svd_getEH(H, self.Ns)
        # EH = H
        # time_end = time.time()
        # print(time_end-time_start)
        EH = EH.reshape(BS, 2, K * self.Ns, Nt)
        v = self.Digital(EH, anhang)
        v = v.reshape(BS, 2, K, self.Ns, Nt)
        return v

class SepTGNN(nn.Module):
    #SEPerated BF and PA policy, but learned Together with a single GNN.
    def __init__(self, hidden_dim, device): ###don't consider concatination here
        super(SepTGNN, self).__init__()
        self.batch_norms1 = torch.nn.ModuleList()

        self.P1 = nn.ParameterList()#只有3个权重
        self.P2 = nn.ParameterList()
        self.P3 = nn.ParameterList()
        self.P4 = nn.ParameterList()

        self.dim = [2] + list(hidden_dim) + [3]
        for i in range(len(self.dim)-2):
            self.batch_norms1.append(nn.BatchNorm2d(self.dim[i + 1]))
        for i in range(len(self.dim) - 1):
            ini = 1.0 / torch.sqrt(torch.FloatTensor([self.dim[i]*self.dim[i+1]]))
            self.P1.append(nn.Parameter(torch.rand([self.dim[i + 1], 2*self.dim[i]], requires_grad=True) * 2 * ini - ini))
            self.P2.append(nn.Parameter(torch.rand([self.dim[i + 1], 2*self.dim[i]], requires_grad=True) * 2 * ini - ini))
            self.P3.append(nn.Parameter(torch.rand([self.dim[i + 1], 2*self.dim[i]], requires_grad=True) * 2 * ini - ini))
            self.P4.append(nn.Parameter(torch.rand([self.dim[i + 1], 2*self.dim[i]], requires_grad=True) * 2 * ini - ini))


        # self.activation = nn.LeakyReLU()
        self.activation = nn.Tanh()
        self.device = device

    def forward(self, H0, Ns, anhang=False):
    #### A(Batch_x):(Batch_size, 2, K , Nt)
        BS, _, KS, Nt = H0.size()
        # time_start = time.time()
        K = KS // Ns
        H0_Re = H0.permute(0, 3, 2, 1)[..., 0:1]; H0_Im = -1 * H0.permute(0, 3, 2, 1)[..., 1:2] #####conjugate
        H0h = torch.concat((H0_Re, H0_Im), dim=3)####conjugate (Batch_size, Nt, K, channel0 = 2)
        H0h = torch.unsqueeze(H0h, dim=1) ### (Batch_size, 1, Nt, K, channel0 = 2)
        D = H0#
        for i in range(len(self.P1)):
            D = D.view(BS, int(D.shape[1]/2), 2, KS, Nt).permute(0,1,3,4,2)
            ####H0h:(Batch_size, 1, Nt, K, 2) ; D:(Batch_size, channel/2, K, Nt, 2)
            Alpha = complex_matmul_model(D, H0h * torch.ones([1, int(D.shape[1]), 1, 1, 1]).to(self.device))#this function is to calculate tensors with channels on the last dim
            ####(Batch_size, channel/2, Nt, Nt, 2)
            B = complex_matmul_model(Alpha,D)/Nt
            C = torch.concat((D, B), dim=-1) ### dim(Batch_size, channel/2, K, Nt, 4)
            C = torch.reshape(C.permute(0, 1, 4, 2, 3), [-1, int(C.shape[1]*C.shape[-1]), KS, Nt])
            D1 = torch.matmul(self.P1[i], C.view([BS, 2*self.dim[i], -1])).view([BS, self.dim[i+1], KS, Nt])
            D2 = torch.matmul(self.P2[i],torch.mean(C,-1).view(BS,2*self.dim[i],-1)).view(BS,self.dim[i+1],KS,1)
            D3 = torch.matmul(self.P3[i],torch.mean(C,-2).view(BS,2*self.dim[i],-1)).view(BS,self.dim[i+1],1,Nt)
            D4 = torch.matmul(self.P4[i],torch.mean(C.view(BS,2*self.dim[i],K,Ns,Nt),-2,keepdim=True).repeat(1,1,1,Ns,1).view(BS,2*self.dim[i],-1)).view(BS, self.dim[i + 1], KS, Nt)

            D = D1 + 0.1*D2 + 0.1*D3 +0.3*D4
            #激活
            if i != len(self.P1)-1:
                D = self.activation(D)
                D = self.batch_norms1[i](D)
                # A = self.activation(A)

        #D:[BS,3,K,Nt]
        V = D[:,0:2,:,:]
        p = torch.mean(D[:,2:3,:,:],dim=-1,keepdim=True) #[BS,1,K,1]
        p = nn.Softplus()(p)+0.005
        pnorm = torch.norm(p, dim=-2, keepdim=True)
        p = torch.where(pnorm != 0, p / pnorm, p)
        return V, p