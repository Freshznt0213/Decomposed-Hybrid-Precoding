function [WBB_cell] = rzf(H_cell, Ns, Pt, sigma)
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


%% 3) RZF 基带预编码
WBB_cell = cell(NRB,K);
for nrb = 1 : NRB
    % RZF 计算：WBB = HBB' * inv(HBB*HBB' + sigma * I)
    HBB = H_miso{nrb};
    A = (HBB * HBB') + sigma*Keq/Pt * eye(Keq);
    WBB = HBB' / A;  % NRF x KNs
    
    % 对整体预编码做归一化（常见做法：使 Frobenius 范数 = sqrt(KNs)）
    if false
        normTotal = norm(WBB, 'fro');
        WBB = WBB*sqrt(Pt/NRB)/normTotal;
    else
         WBB = WBB ./ vecnorm(WBB);
         Heq = diag(HBB*WBB);
         p = diag(waterfilling(Heq,Pt,sigma));
         WBB = WBB * sqrt(p);
    end
    WBB_cell_single = mat2cell(WBB, Nt, Ns * ones(K, 1));

    WBB_cell(nrb,:) = WBB_cell_single;
end
end

