# coding=UTF-8
import numpy as np
import scipy.linalg as la
import os
import sys
import time
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch


def generate_H_rayleigh(*args):#rayleigh channel  real and imag gaussian
    H = 1 / np.sqrt(2)  * (np.random.randn(*args).astype(np.float32) + 1j * np.random.randn(*args).astype(np.float32))

    return H

def spreadAoD(mu, std): #laplacian distribution 
    b=std/np.sqrt(2)
    a=np.random.rand()-0.5 #probability related to CDF
    x=mu-b*np.sign(a)*np.log(1-2*abs(a)) #value corresponding to probability a
    return x


def generate_H_SV(K,Nt,Ncl,Nray,d_lamda=0.5,beta=1,std=10/180*np.pi): #MISO SV channel model
    Ct = np.arange(Nt) 
    H = np.zeros([K,Nt],dtype=complex)
    for k in range(K): #for each user
        Htemp = np.zeros([1,Nt],dtype=complex)
        for ii in range(Ncl): #cluster
            fhi_i = np.random.uniform(0,2*np.pi)
            for jj in range(Nray): #ray
                a = (np.random.randn()+1j*np.random.randn())/np.sqrt(2)#random amplitude
                fhi_ij = spreadAoD(fhi_i,std)
                ft = 1 / np.sqrt(Nt) * np.exp(Ct * 1j * 2 * np.pi * d_lamda * np.sin(fhi_ij))
                Htemp = Htemp+ a*ft
        H[k] = Htemp
    H = H * np.sqrt(Nt/Ncl/Nray) #cluster and ray number normalization
    return H


def generate_HMIMO_SV(K,Nt,Nr,Ncl,Nray,d_lamda=0.5,beta=1,std=10/180*np.pi, islarge=False):
    Ct = np.arange(Nt).reshape(1,Nt)
    Cr = np.arange(Nr).reshape(Nr,1)
    H = np.zeros([K,Nr,Nt],dtype=complex)
    dmin = 50
    dmax = 200
    fc = 28 #GHz
    for k in range(K):
        Htemp = np.zeros([Nr,Nt],dtype=complex)
        for ii in range(Ncl):
            fhi_i = np.random.uniform(0,2*np.pi)
            the_i = np.random.uniform(0,2*np.pi)
            for jj in range(Nray):
                a = (np.random.randn()+1j*np.random.randn())/np.sqrt(2)
                fhi_ij = spreadAoD(fhi_i,std)
                ft_Nt = 1 / np.sqrt(Nt) * np.exp(Ct * 1j * 2 * np.pi * d_lamda * np.sin(fhi_ij))
                b = (np.random.randn()+1j*np.random.randn())/np.sqrt(2)
                the_ij = spreadAoD(the_i,std)
                ft_Nr = 1 / np.sqrt(Nr) * np.exp(Cr * 1j * 2 * np.pi * d_lamda * np.sin(the_ij))
                plusterm = np.matmul(b*ft_Nr,a*ft_Nt)
                Htemp = Htemp+ plusterm
        if islarge:
            dis = np.sqrt(dmin ** 2 + np.random.rand(1) * (dmax ** 2 - dmin ** 2))
            PL = 13.54 + 39.08 * np.log10(dis) + 20 * np.log10(fc)
            Htemp = Htemp * np.sqrt(1 / (10 ** (PL / 10)))

        H[k,:,:] = Htemp
    H = H * np.sqrt(Nt*Nr/Ncl/Nray)

    return H


def generate_H_dataset(type, number, K, Nt, Nr, K_Factor_dB, Ncl=4, Nray=5): #if MISO, Nr = 1
    #生成训练的数据集
    start = time.time()
    H = np.zeros([number, 2, K, Nr, Nt],dtype=np.float32)
    for i in range(number):
        if i % 2000 == 0:
            end = time.time()
            print(i,end-start)
            sys.stdout.flush()
            start = time.time()
        if type == 'SV':
            Htemp = generate_HMIMO_SV(K, Nt, Nr, Ncl, Nray)
            channelmodel = '-'.join(["SV",str(Ncl),str(Nray)])
        elif type == 'SVl':
            Htemp = generate_HMIMO_SV(K, Nt, Nr, Ncl, Nray, islarge=True)
            channelmodel = '-'.join(["SVl", str(Ncl), str(Nray)])
        elif type == 'Rayleigh':
            Htemp = generate_H_rayleigh(K, Nr, Nt)
            channelmodel = "Rayleigh"

        H[i,0] = np.real(Htemp)
        H[i,1] = np.imag(Htemp)

    if not os.path.exists("CHdata/"):
        os.makedirs("CHdata/")
    np.save("./CHdata/CH_K" + str(K) + "_Nt" + str(Nt) + "_Nr" + str(Nr)  + "_number" + str(number) + '_' + channelmodel + ".npy", H)
    
    return H

if __name__ == '__main__':
    K = 4
    Nt = 16
    Nr = 1


    Ncl = 4
    Nray = 5

    K_Factor_dB = 10
    #产生数据集
    generate_H_dataset('SV',4000, K, Nt, Nr, K_Factor_dB, Ncl=Ncl, Nray=Nray)
    generate_H_dataset('SV',2000, K, Nt, Nr, K_Factor_dB, Ncl=Ncl, Nray=Nray)

