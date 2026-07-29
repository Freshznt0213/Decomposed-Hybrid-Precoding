function H_rec = recover_cell_from_horzcat(H_big, NRB, K, Ns)
%RECOVER_CELL_FROM_HORZCAT Recover [NRB,K] cell from horzcat matrix
%
% H_big : [Nt, NRB*K*Ns] complex matrix
% H_rec : [NRB, K] cell, each cell [Nt, Ns]

    H_rec = cell(NRB, K);

    idx = 0;
    for k = 1:K
        for r = 1:NRB
            idx = idx + 1;
            col_idx = (idx-1)*Ns + (1:Ns);
            H_rec{r,k} = H_big(:, col_idx);
        end
    end
end


