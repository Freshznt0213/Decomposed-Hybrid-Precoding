import torch
import torch.nn as nn
import numpy as np
from random import shuffle
import time
import sys
import math

def complex_matmul_model(A, B):
    ###channel put on last dim
    return torch.cat([torch.unsqueeze(torch.matmul(A[..., 0], B[..., 0]) - torch.matmul(A[..., 1], B[..., 1]), dim=-1),
                      torch.unsqueeze(torch.matmul(A[..., 0], B[..., 1]) + torch.matmul(A[..., 1], B[..., 0]), dim=-1)],
                     dim=-1)

def complex_hadma_model(A,B):
    return torch.cat([torch.unsqueeze(A[..., 0]*B[..., 0] - A[..., 1]*B[..., 1], dim=-1),
                          torch.unsqueeze(A[..., 0]*B[..., 1] + A[..., 1]*B[..., 0], dim=-1)],dim=-1)

def complex_norm2_model(A):
    return torch.sqrt(A[..., 0]**2 + A[..., 1]**2)


def complex_conjT_model(A):
    ###channel put on last dim
    return torch.cat([A[..., 0:1], -A[..., 1:2]], dim=-1).transpose(-2,-3)

def complex_conjT(A):
    ###channel put on second dim
    return torch.cat([A[:,0:1], - A[:,1:2]], dim=1).transpose(-1,-2)

def complex_conj(A):
    ###channel put on second dim
    return torch.cat([A[:,0:1], - A[:,1:2]], dim=1)


def complex_matmul(A, B):
    return torch.cat([torch.unsqueeze(torch.matmul(A[:, 0], B[:, 0]) - torch.matmul(A[:, 1], B[:, 1]),dim = 1),
                      torch.unsqueeze(torch.matmul(A[:, 0], B[:, 1]) + torch.matmul(A[:, 1], B[:, 0]),dim = 1)],
                     dim=1)

def complex_hadma(A, B):
    return torch.cat([torch.unsqueeze(A[:, 0]*B[:, 0] - A[:, 1]*B[:, 1],dim = 1),
                      torch.unsqueeze(A[:, 0]*B[:, 1] + A[:, 1]*B[:, 0],dim = 1)],
                     dim=1)


def replicate_diagonal(matrix, k):
    '''
    :param matrix: original matrix wished to be duplicated
    :param k: times for duplication
    :return: a block diagonal matrix
    '''
    diag_blocks = [matrix] * k
    result_matrix = torch.block_diag(*diag_blocks)
    return result_matrix


def expand_square(matrix, k):
    n = matrix.size(0)
    expanded_matrix = matrix.unsqueeze(0).unsqueeze(1)
    replicated_diagonal = expanded_matrix.repeat(k, k, 1, 1).permute(0, 2, 1, 3).contiguous()
    return replicated_diagonal.view(n * k, n * k)

def ampli2(A):
    # BCHW表示复矩阵
    # 结果没有C轴
    return A[:, 0, :, :] ** 2 + A[:, 1, :, :] ** 2

def normuni(A):
    A_norm = torch.norm(A, dim=1, keepdim=True)
    return A / A_norm

def normuni2(A):
    [BS,C,H,W] = A.size()
    A_trans = A.reshape(BS,2,C//2,H,W)
    A_trans_norm = torch.norm(A_trans, dim=1, keepdim=True)
    A_deno = A_trans / A_trans_norm
    A_deno = A_deno.reshape(BS,C,H,W)
    return A / A_deno

def comp_svd_getEH(H0,Ns):
    BS, _, K, Nr, Nt = H0.size()
    H0_comp = torch.complex(H0[:, 0], H0[:, 1]).reshape(BS * K, Nr, Nt)  # BS,K,Nr,Nt
    U, sigma, V = torch.linalg.svd(H0_comp,full_matrices=False)  # U:[BS*K,Nr,Nr],sigma:[BS*K,Nr],V:[BS*K,Nt,Nt]
    V = V[:, 0:Ns, :]
    sigma = sigma[:, 0:Ns].unsqueeze(-1)
    # sigma_pro = torch.log(1+sigma)+0.3
    V = V * sigma * torch.sqrt(torch.tensor(Ns/Nr, dtype=torch.float32))
    V = torch.concat([torch.real(V).unsqueeze(1), torch.imag(V).unsqueeze(1)], dim=1)  # [BS*K,2,Ns,Nt]
    EH = V.reshape(BS, K, 2, Ns, Nt).permute(0, 2, 1, 3, 4)

    return EH

def dft_matrix(N):
    # 生成行和列索引
    k = torch.arange(N).reshape(N, 1)
    n = torch.arange(N).reshape(1, N)

    # 计算DFT矩阵的复数部分
    omega = torch.exp(-2j * math.pi * k * n / N)

    # 归一化因子
    F = omega / math.sqrt(N)
    F = torch.cat([torch.real(F).unsqueeze(0),torch.imag(F).unsqueeze(0)],dim=0)
    return F

def random_unitary_matrix(N):

    # complex Gaussian matrix
    real = torch.randn(N, N)
    imag = torch.randn(N, N)
    A = real + 1j * imag

    # QR decomposition
    Q, R = torch.linalg.qr(A)

    # phase normalization
    diag = torch.diagonal(R)
    phase = diag / torch.abs(diag)
    Q = Q * phase.conj()

    # convert to real-imag representation
    Q_real = torch.real(Q).unsqueeze(0)
    Q_imag = torch.imag(Q).unsqueeze(0)

    Q_out = torch.cat([Q_real, Q_imag], dim=0)

    return Q_out

