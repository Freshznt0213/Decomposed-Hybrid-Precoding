function [WRF, WBB, Hstack] = beam_sel_rzf(H, Ns, NRF, sigma)
% hybrid_precoding_from_H_rsrp  基于 RSRP 归一化波束选择 的混合预编码函数
% 输入:
%   H    - 原始信道，维度 [K, Nr, Nt]
%   Ns   - 每用户保留的奇异分量数（<= Nr）
%   NRF  - 要选的射频链/波束数（<= Nt）
%   sigma- 噪声功率（标量），用于 RZF 正则项
% 输出:
%   WRF  - RF (analog) 预编码矩阵，尺寸 [Nt, NRF]
%   WBB  - baseband (digital) 预编码矩阵，尺寸 [NRF, K*Ns]
%   H_tilde - 等效信道，尺寸 [K, Ns, Nt]
%   idx_selected - 被选择的码本列索引（1..Nt），长度 NRF
%
% 说明:
%   波束选择采用：对每个等效流在码本列上做归一化（行归一化），
%   然后对所有流求和得到每列的得分，挑得分最高的 NRF 列。

%% 参数检查
if nargin < 4
    error('需要 4 个输入参数：H, Ns, NRF, sigma');
end
[K, Nr, Nt] = size(H);
if Ns > Nr
    error('Ns 不能大于 Nr');
end
if NRF > Nt
    error('NRF 不能大于 Nt');
end

%% 1) 逐用户 SVD 并构造 H_tilde (K x Ns x Nt)
H_tilde = zeros(K, Ns, Nt); % 保持维度，不用 squeeze 导致丢维
for k = 1:K
    Hk = squeeze(H(k,:,:)); % Hk 尺寸 [Nr, Nt]
    [Uk, Sk, Vk] = svd(Hk, 'econ');
    for s = 1:Ns
        si = Sk(s,s);
        vi = Vk(:,s);           % Nt x 1
        H_tilde(k, s, :) = (si * (vi.'));  % 1 x Nt (复共轭转置考虑下面一致性)
    end
end

%% 将 H_tilde 展成 [K*Ns, Nt] 的矩阵（按用户 then stream 顺序）
KNs = K * Ns;
Hstack = reshape(permute(H_tilde, [1,2,3]), [KNs, Nt]); 
% 现在 Hstack 的每一行对应一个等效流（row i is stream i）

%% 2) 波束扫描（借鉴 RSRP 归一化流程）
% 生成 Nt x Nt DFT 码本（列单位范数）
F = dftmtx(Nt);                % Nt x Nt
codebook = (1/sqrt(Nt)) * F;   % 归一化列向量
Ncb = Nt; % 码本列数

% 计算每个等效流在每个码本列上的复内积 s = Hstack * codebook (KNs x Ncb)
s = Hstack * codebook;              % KNs x Ncb
% RSRP 矩阵：每个元素为功率
RSRP_matrix = abs(s).^2;            % KNs x Ncb

% 行归一化：每个等效流在码本维度求和然后除（避免除零）
row_sums = sum(RSRP_matrix, 2);     % KNs x 1
% 防止除以 0
zero_rows = (row_sums == 0);
row_sums(zero_rows) = 1;
RSRP_normalized = RSRP_matrix ./ row_sums;  % KNs x Ncb

% 对所有等效流在行方向求和，得到每个码本列的总体得分（1 x Ncb）
powerVec = sum(RSRP_normalized, 1);  % 1 x Ncb

% 选择得分最高的 NRF 列
[~, idx_sorted] = sort(powerVec, 'descend');
idx_selected = idx_sorted(1:NRF);

% 构造 WRF（Nt x NRF）
WRF = codebook(:, idx_selected);   % Nt x NRF

%% 3) RZF 基带预编码
HBB = Hstack * WRF;  % [KNs, NRF]

% RZF 计算：WBB = HBB' * inv(HBB*HBB' + sigma * I)
A = (HBB * HBB') + sigma * eye(KNs);
WBB = HBB' / A;  % NRF x KNs

% 对整体预编码做归一化（常见做法：使 Frobenius 范数 = sqrt(KNs)）
W_total = WRF * WBB; % Nt x KNs
normTotal = norm(W_total, 'fro');
if normTotal == 0
    warning('总预编码矩阵范数为 0，跳过归一化');
else
    scale = 1 / normTotal;
    WBB = WBB * scale;
end

end
