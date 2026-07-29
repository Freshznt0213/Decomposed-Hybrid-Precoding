function [ FRF,FBB ] = MO( Fopt, NRF, Pt )
%need MATLAB toolbox: Manopt
    [Nt, ~] = size(Fopt);
    y = [];
    FRF = exp( 1i*unifrnd(0,2*pi,Nt,NRF) );
    while(isempty(y) || abs(y(1)-y(2))>5e-4)
        FBB = pinv(FRF) * Fopt;
        y(1) = norm(Fopt - FRF * FBB, 'fro')^2;
        [FRF, y(2)] = sig_manif(Fopt, FRF, FBB);
    end
FBB = sqrt(Pt) * FBB / norm(FRF * FBB, 'fro');
end