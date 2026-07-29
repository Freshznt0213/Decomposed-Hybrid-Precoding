clear
clc
K = 4;%用户数
NRF = 8;%射频链数
Nt = 16;%基站天线数
Nr = 1;%用户接收天线数
Ns = 1;
K_Factor_dB = 10;
K_Factor = 10^(K_Factor_dB/10);
Ncl = 4;% SV parameter
Nray = 5;% SV parameter

SNR = 10;%
sigma2 = 10^(-SNR/10);

Time = 10;

Rsum1 = zeros(Time,1);
Rsum2 = zeros(Time,1);
channel_set = cell(Time,1);
WRF_set = cell(Time,1);
sumtime1 = 0;
sumtime2 = 0;
for ti = 1:Time
    ti
    %---------------channel realization--------------%
    % H = Rayleigh_corr(K, Nr, Nt,0.4, 0.4);%瑞利信道
    H = SVMIMOmodel(K,Nt,Nr,Ncl,Nray,10/180*pi);%毫米波信道 10/180*pi是AoD拉普拉斯分布的标准差
    % channel_set{ti} = H;
    % ---------precoding and rate calculating----------%
    % tic
    W_wmmse = MIMOWMMSE(H,Ns,SNR);%return a cell
    time1 = toc;
    sumtime1 = sumtime1 + time1;
    Rsum1(ti) = MIMOcalrate(H,W_wmmse,SNR);
 
    W_wmmse_stack = horzcat(W_wmmse{:});
    tic
    [FRF,FBB] = MO(W_wmmse_stack,NRF,1);%混合MO方法 一般最好，最慢
    time2 = toc;
    sumtime2 = sumtime2 + time2;
    W_HBD = FRF*FBB;
    W_HBD_cell = mat2cell(W_HBD, Nt, Ns * ones(K, 1));
    % WRF_set{ti} = FRF;
    Rsum2(ti) = MIMOcalrate(H,W_HBD_cell,SNR);
    % tic
    % [WRF,WBB,Hsvd] = beam_sel_rzf(H, Ns, NRF, sigma2);%混合MO方法 一般最好，最慢
    % time2 = toc;
    % sumtime2 = sumtime2 + time2;
    % W_HBD = WRF*WBB;
    % Rsum2(ti) = calrate(Hsvd,W_HBD,K*Ns,SNR);
    % tic
    % [WBB,Hsvd] = rzf(H, Ns, sigma2);%混合MO方法 一般最好，最慢
    % time3 = toc;
    % sumtime3 = sumtime3 + time3;
    % Rsum3(ti) = calrate(Hsvd,WBB,K*Ns,SNR);
    % tic
    % [WRF,WBB,Hsvd] = WMH_rzf(H, Ns, NRF, sigma2);%混合MO方法 一般最好，最慢
    % time2 = toc;
    % sumtime2 = sumtime2 + time2;
    % Rsum2(ti) = calrate(Hsvd,WRF*WBB,K*Ns,SNR);
end

WMMSE_rate = sum(Rsum1)/Time
MO_rate = sum(Rsum2)/Time
% stage1 = sumtime1/Time*1000
% stage2 = sumtime2/Time*1000

% abs(FRF)%检查约束
% sum(sum(abs(FRF*FBB).^2))