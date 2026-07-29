function [Rsum] = MIMOcalrate(H,V,SNR)
sigma2=10^(-SNR/10);
[K,Nr,Nt] = size(H);
H_cell = cell(K, 1);
Rsum = 0;
% convert channel tensor to channel cell
for k = 1:K
    H_cell{k} = reshape(H(k,:,:), [Nr, Nt]);
end
for m = 1:K
    temp = zeros(Nr);
    for i = 1:K
        if i ~= m
            temp = temp + H_cell{m}*V{i}*V{i}'*H_cell{m}';
        end
    end
    temp = temp + sigma2*eye(Nr);
    SINRtemp = H_cell{m}*V{m}*V{m}'*H_cell{m}'/temp;
    Rtemp = log2(abs(det(eye(Nr)+SINRtemp)));
    Rsum = Rsum + Rtemp;
end
end

