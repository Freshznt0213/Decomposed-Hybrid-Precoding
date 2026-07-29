function [SE] = MIMOcalSE(H_cell,V,sigma2)
[NRB, K] = size(H_cell);
[Nr, Nt] = size(H_cell{1});
SE = 0;
for nrb = 1:NRB
    H_cell_nrb = H_cell(nrb,:);
    V_nrb = V(nrb,:);
    Rsum = 0;
    for m = 1:K
        temp = zeros(Nr);
        for i = 1:K
            if i ~= m
                temp = temp + H_cell_nrb{m}*V_nrb{i}*V_nrb{i}'*H_cell_nrb{m}';
            end
        end
        temp = temp + sigma2*eye(Nr);
        SINRtemp = H_cell_nrb{m}*V_nrb{m}*V_nrb{m}'*H_cell_nrb{m}'/temp;
        Rtemp = log2(abs(det(eye(Nr)+SINRtemp)));
        Rsum = Rsum + Rtemp;
    end
    SE = SE + Rsum;
end
SE = SE / NRB;


