function [WRF, WBB_cell] = greedy_sel_rzf(H_cell, Ns, NRF, Pt, sigma)
[NRB, K] = size(H_cell);
[Nr, Nt] = size(H_cell{1});
Keq = K*Ns;

%% 1) 逐用户 SVD 并构造 H_tilde (NRB x K x Ns x Nt)
H_tilde = cell(NRB, K);
for nrb = 1 : NRB
    for k = 1:K
        Hk = squeeze(H_cell{nrb,k}); % Hk 尺寸 [Nr, Nt]
        [Uk, Sk, Vk] = svd(Hk, 'econ');
        si = Sk(1:Ns,1:Ns);
        vi = Vk(:,1:Ns);           % Nt x 1
        H_tilde{nrb,k} = (si * (vi'));  % 1 x Nt (复共轭转置考虑下面一致性)
    end
end

H_miso = cell(NRB,1);
for r = 1:NRB
    H_miso{r} = vertcat(H_tilde{r, :});
end

%% 将 H_tilde 展成 [NRB*K*Ns, Nt] 的矩阵
M = NRB*K*Ns;
Hstack = vertcat(H_miso{:}); 
% 现在 Hstack 的每一行对应一个等效流

%% 2) 贪婪波束选择
% 生成 Nt x Nt DFT 码本（列单位范数）
Ant_mode = 'UPA';
switch Ant_mode
    case 'ULA'
        F = dftmtx(Nt);                % Nt x Nt
        codebook = (1/sqrt(Nt)) * F;   % 归一化列向量
        Ncb = Nt; % 码本列数
    case 'UPA'
        n = log2(Nt/2);
        M = 2^floor(n/2); N = 2^ceil(n/2);
        Fv = dftmtx(M) / sqrt(M);
        Fh = dftmtx(N) / sqrt(N);
        Fspa = kron(Fh, Fv);
        
        F = kron(eye(2), Fspa);   % Nt x Nt
        codebook = F;
        Ncb = Nt;
end
Pi = Hstack;
WRF_matrix = zeros(Nt, 0);  % N×0 空矩阵
for ii = 1:NRF
    Phi = Pi * codebook; % K * NCB
    candi = vecnorm(Phi);
    max_val = max(candi);      % 获取最大值
    max_idx = find(candi == max_val);  % 获取所有最大值位置
    picked = codebook(:, max_idx);
    WRF_matrix = horzcat(WRF_matrix, picked);
    Pi = Pi * (eye(Nt)-WRF_matrix*WRF_matrix');
end
% 构造 WRF（Nt x NRF）
WRF = WRF_matrix;   % Nt x NRF

%% 3) RZF 基带预编码
WBB_cell = cell(NRB,K);
for nrb = 1 : NRB
    % RZF 计算：WBB = HBB' * inv(HBB*HBB' + sigma * I)
    HBB = H_miso{nrb}*WRF;
    A = (HBB * HBB') + sigma*Keq/Pt * eye(Keq);
    WBB = HBB' / A;  % NRF x KNs
    
    % 对整体预编码做归一化（常见做法：使 Frobenius 范数 = sqrt(KNs)）
    % W_total = WRF * WBB; % Nt x KNs
    if false
        normTotal = norm(WBB, 'fro');
        WBB = WBB*sqrt(Pt)/normTotal;
    else
         WBB = WBB ./ vecnorm(WBB);
         Heq = diag(HBB*WBB);
         p = diag(waterfilling(Heq,Pt,sigma));
         WBB = WBB * sqrt(p);
    end
    % WBB_cell_single = cell(K,1);
    % 
    % for k = 1:K
    %     cols = (k-1)*Ns + (1:Ns);
    %     WBB_cell_single{k} = WBB(:, cols);
    % end
    WBB_cell_single = mat2cell(WBB, NRF, Ns * ones(K, 1));

    WBB_cell(nrb,:) = WBB_cell_single;
end
end

