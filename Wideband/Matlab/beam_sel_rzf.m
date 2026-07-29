function [WRF, WBB_cell] = beam_sel_rzf(H_cell, Ns, NRF, Pt, sigma)
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

%% 2) 波束扫描（借鉴 RSRP 归一化流程）
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
        Fspa = kron(Fv, Fh);
        
        F = kron(eye(2), Fspa);   % Nt x Nt
        codebook = F;
        Ncb = Nt;
end
% 计算每个等效流在每个码本列上的复内积 s = Hstack * codebook (M x Ncb)
s = Hstack * codebook;              % M x Ncb
% RSRP 矩阵：每个元素为功率
RSRP_matrix = abs(s).^2;            % M x Ncb

% 行归一化：每个等效流在码本维度求和然后除（避免除零）
row_sums = sum(RSRP_matrix, 2);     % M x 1
% 防止除以 0
zero_rows = (row_sums == 0);
row_sums(zero_rows) = 1;
RSRP_normalized = RSRP_matrix ./ row_sums;  % M x Ncb

% 对所有等效流在行方向求和，得到每个码本列的总体得分（1 x Ncb）
powerVec = sum(RSRP_normalized, 1);  % 1 x Ncb

% 选择得分最高的 NRF 列
[~, idx_sorted] = sort(powerVec, 'descend');
idx_selected = idx_sorted(1:NRF);

% 构造 WRF（Nt x NRF）
WRF = codebook(:, idx_selected);   % Nt x NRF

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
