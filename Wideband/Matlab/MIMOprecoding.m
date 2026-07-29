% single cell Wideband MU-MIMO Precoding
% simulate a cell (a BS) or a sector in a cell
% consider large scale fading
% Output SE
% ZNT Dec. 2025
clear
clc

K = 8;%用户数
NRF = 48;%射频链数
NRB = 16;
Nt = 1024;%基站天线数
Nr = 4;%用户接收天线数
Ns = 4;
Ncl = 4;% SV parameter
Nray = 5;% SV parameter

is_large_scale = true;
num_dataset = 100;

%% Read Channel
if is_large_scale
    %% Transmit Power
    N_subcarrier = 3168;
    Pt_dBm = 35;
    Pt_sub = 10^((Pt_dBm - 30)/10);
    Pt_W = Pt_sub * NRB;
    
    %% Noise Power
    fc = 28;
    BW = 100e6;
    sigma2_dBm = -174 + 10*log10(BW) + 9;
    sigma2_W = 10^((sigma2_dBm - 30)/10);
    folder = fullfile('..','Dataset', ...
        sprintf('Test_WBUma_fc%d_NRBG%d_K%d_Nt%d_Nr%d_num%d', ...
        fc, NRB, K, Nt, Nr, num_dataset));
else
    SNR = 0;
    Pt_sub = 1;
    Pt_W = Pt_sub * NRB;
    sigma2_W = 10^(-SNR/10);
    folder = fullfile('..','Dataset', ...
    sprintf('Test_WBSV_NRBG%d_K%d_Nt%d_Nr%d_Ncl%d_Nray%d', ...
    NRB, K, Nt, Nr, Ncl, Nray));
end

load(fullfile(folder, 'H_tensor.mat'));
% CH_all = h5read(fullfile(folder, 'H_tensor.mat'), '/CH_all');

Time = 10;

Rsum1 = zeros(Time,1);
Rsum2 = zeros(Time,1);
Rsum3 = zeros(Time,1);
Rsum4 = zeros(Time,1);
channel_set = cell(Time,1);
WRF_set = cell(Time,1);
sumtime1 = 0;
sumtime2 = 0;
sumtime3 = 0;
sumtime4 = 0;

for ti = 1:Time
    ti
    %---------------channel realization--------------%
    H_real = squeeze(CH_all(ti,1,:,:,:,:));
    H_imag = squeeze(CH_all(ti,2,:,:,:,:));
    H = H_real + 1j*H_imag;
    H_cell = cell(NRB, K);
    % convert channel tensor to channel cell
    for nrb = 1: NRB
        for k = 1:K
            H_cell{nrb,k} = reshape(H(nrb,k,:,:), [Nr, Nt]);
        end
    end
    % ---------precoding and rate calculating----------%
    tic
    W_rzf = rzf(H_cell, Ns, Pt_sub, sigma2_W);%混合MO方法 一般最好，最慢
    time1 = toc;
    sumtime1 = sumtime1 + time1;
    Rsum1(ti) = MIMOcalSE(H_cell,W_rzf,sigma2_W);

    % tic
    % W_wmmse = MIMOWMMSE(H_cell,W_rzf,Ns,Pt_sub,sigma2_W);%return a cell
    % time2 = toc;
    % sumtime2 = sumtime2 + time2;
    % Rsum2(ti) = MIMOcalSE(H_cell,W_wmmse,sigma2_W);

    tic
    [WRF,WBB] = greedy_sel_rzf(H_cell, Ns, NRF, Pt_sub, sigma2_W);%混合MO方法 一般最好，最慢
    time3 = toc;
    sumtime3 = sumtime3 + time3;
    WBB_stack = horzcat(WBB{:});
    W_HBD = WRF*WBB_stack;
    W_HBD_cell = recover_cell_from_horzcat(W_HBD, NRB, K, Ns);
    Rsum3(ti) = MIMOcalSE(H_cell,W_HBD_cell,sigma2_W);
 
    % W_wmmse_stack = horzcat(W_wmmse{:});
    % tic
    % [FRF,FBB] = MO(W_wmmse_stack,NRF,Pt_W);%混合MO方法 一般最好，最慢
    % time4 = toc;
    % sumtime4 = sumtime4 + time4;
    % W_HBD = FRF*FBB;
    % W_HBD_cell = recover_cell_from_horzcat(W_HBD, NRB, K, Ns);
    % Rsum4(ti) = MIMOcalSE(H_cell,W_HBD_cell,sigma2_W);
end

rzf_rate = sum(Rsum1)/Time
WMMSE_rate = sum(Rsum2)/Time

Beamrzf_rate = sum(Rsum3)/Time
MO_rate = sum(Rsum4)/Time

% stage1 = sumtime1/Time*1000
% stage2 = sumtime2/Time*1000
% mean(sum(sum(abs(CH_all).^2, 2), 6),'all')
% Pt_W * mean(sum(sum(sum(abs(CH_all).^2, 2), 6), 5),'all') / sigma2_W
% mean(sum(sum(sum(abs(CH_all).^2, 2), 6), 5),'all')
% abs(FRF)%检查约束
% sum(sum(abs(FRF*FBB).^2))

% CH_comp = squeeze(CH_all(:,1,:,:,:,:)+CH_all(:,2,:,:,:,:));
% CH_comp_sample = squeeze(CH_comp(22,:,:,:,:));
% CH_comp_RBG1 = squeeze(CH_comp_sample(1,:,1,:));
% CH_comp_RBG2 = squeeze(CH_comp_sample(2,:,1,:));
% C = CH_comp_RBG1*CH_comp_RBG2';
% normA = vecnorm(CH_comp_RBG1, 2, 2);  % 2-范数，沿第2维计算（按行）
% normB = vecnorm(CH_comp_RBG2, 2, 2);
% normMatrix = normA * normB';
% C_normalized = C ./ normMatrix;