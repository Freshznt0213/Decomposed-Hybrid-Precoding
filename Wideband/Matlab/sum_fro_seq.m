function val = sum_fro_seq(cellMat)
%SUM_FRO_SQ Sum of squared Frobenius norms over a cell array
%
%   cellMat: [NRB, K] cell, each cell contains a [Nt, Ns] complex matrix
%   val    : scalar, sum_{r,k} ||cellMat{r,k}||_F^2

    val = 0;

    for idx = 1:numel(cellMat)
        X = cellMat{idx};
        % Frobenius norm squared
        val = val + sum(abs(X(:)).^2);
    end
end
