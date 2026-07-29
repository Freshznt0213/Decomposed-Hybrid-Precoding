function [WBB, Hstack] = rzf(H, Ns, sigma)
% hybrid_precoding_from_H  基于输入信道 H 生成 RF 和 baseband 预编码
%
% 输入:
%   H    - 原始信道，维度 [K, Nr, Nt]
%   Ns   - 每用户保留的奇异分量数（<= Nr）
%   NRF  - 要选的射频链/波束数（<= Nt）
%   sigma- 噪声功率（标量），用于 RZF 正则项
%
% 输出:
%   WRF  - RF (analog) 预编码矩阵，尺寸 [Nt, NRF]
%   WBB  - baseband (digital) 预编码矩阵，尺寸 [NRF, K*Ns]
%   H_tilde - 等效信道，尺寸 [K, Ns, Nt] （可选）
%
% 说明:
%   - 对每个用户的 [Nr x Nt] 信道做 SVD，保留前 Ns 个奇异值与对应的右奇异向量，
%     并将每个分量以 (s_i * v_i^H) 的形式存为等效向量（长度 Nt）。
%   - 生成 Nt x Nt 的 DFT 码本（归一化），对所有 K*Ns 个等效分量进行波束扫描，
%     以得到每个码本列的累加功率并选取前 NRF 个作为 WRF 的列。
%   - 在 RF 之后得到等效基带信道 HBB = Hstack * WRF (尺寸 [K*Ns, NRF])，
%     用 RZF 计算 WBB = HBB' * inv(HBB*HBB' + sigma*I), 并对总发射功率做尺度归一化。
%
% 注意: 代码中避免使用 squeeze，以确保 Ns=1 时维度不丢失。

%% 参数检查
if nargin < 3
    error('需要 3 个输入参数：H, Ns, sigma');
end
[K, Nr, Nt] = size(H);
if Ns > Nr
    error('Ns 不能大于 Nr');
end

%% 1) 逐用户 SVD 并构造 H_tilde (K x Ns x Nt)
H_tilde = zeros(K, Ns, Nt); % 不用 squeeze，保留维度信息
for k = 1:K
    Hk = squeeze(H(k,:,:)); % Hk 尺寸 [Nr, Nt]，这里的 squeeze 是安全的（从 [1,Nr,Nt] -> [Nr,Nt]）
    % SVD
    [Uk, Sk, Vk] = svd(Hk, 'econ'); % Vk: Nt x min(Nr,Nt)
    % 保留前 Ns 个奇异值分量 (若 Vk 列数 >= Ns)
    for s = 1:Ns
        si = Sk(s,s);
        vi = Vk(:,s);           % Nt x 1
        % 将 si * vi^H 作为等效行向量（1 x Nt）
        H_tilde(k, s, :) = (si * (vi.')) ;  % 保留为实/复行向量形式
    end
end

%% 将 H_tilde 展成 [K*Ns, Nt] 的矩阵（按用户 then stream 顺序）
KNs = K * Ns;
Hstack = reshape(permute(H_tilde, [1,2,3]), [KNs, Nt]);
% 说明：permute 保证了 reshape 后的顺序是按 k 然后 s 的顺序排列

%% 3) RZF 预编码：构造 HBB = Hstack * WRF (KNs x NRF)，然后计算 WBB
HBB = Hstack;  % [KNs, NRF]

% RZF: WBB = HBB' * inv(HBB * HBB' + sigma * I_KNs)
% 为数值稳定性，用 MATLAB 的右除 \ 或 inv 推荐写法：
A = (HBB * HBB') + sigma * eye(KNs);
% 计算 WBB (NRF x KNs)
WBB = HBB' / A;  % 等价于 HBB' * inv(A)
check = HBB*WBB;
% 可选：对总预编码矩阵做归一化（使得总发射功率 = KNs）
% 总预编码矩阵为 W = WRF * WBB (Nt x KNs)
normTotal = norm(WBB, 'fro');
if normTotal == 0
    warning('总预编码矩阵范数为 0，跳过归一化');
else
    scale = 1 / normTotal;
    WBB = WBB * scale;
    % 更新 W_total 如果需要：
    % W_total = W_total * scale;
end
end


