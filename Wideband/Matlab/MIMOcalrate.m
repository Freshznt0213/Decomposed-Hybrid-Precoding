function [Rsum] = MIMOcalrate(H_cell,V,sigma2)
%narrow-band function
[~, K] = size(H_cell);
[Nr, Nt] = size(H_cell{1});
Rsum = 0;

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

