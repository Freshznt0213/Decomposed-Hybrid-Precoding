function [y, cost] = sig_manif(Fopt, FRF, FBB)
[Nt, NRF] = size(FRF);

manifold = complexcirclefactory(Nt*NRF);
problem.M = manifold;

% problem.cost  = @(x) norm( Fopt - reshape(x,Nt,NRF) * FBB,'fro')^2;
% problem.egrad = @(x) -2 * kron(conj(FBB), eye(Nt)) * (Fopt(:) - kron(FBB.', eye(Nt)) * x);
f = Fopt(:);
A = kron(FBB.', eye(Nt));

problem.cost  = @(x) (f-A*x)'*(f-A*x);
problem.egrad = @(x) -2*A'*(f-A*x);

% checkgradient(problem);
warning('off', 'manopt:getHessian:approx');
options.verbosity = 0;
[x,cost,info,options] = conjugategradient(problem,FRF(:), options);
% [x,cost,info,options] = trustregions(problem, FRF(:));
% info.iter
y = reshape(x,Nt,NRF);

end

% function [y, cost] = sig_manif(Fopt, FRF, FBB)
% [Nt, NRF] = size(FRF);
% K = size(FBB,3);
% 
% manifold = complexcirclefactory(Nt*NRF);
% problem.M = manifold;
% 
% %parfor k = 1:K
% for k = 1:K    
%     temp = Fopt(:,:,k);
%     A = kron(FBB(:,:,k).', eye(Nt));
%     C1(:,:,k) = temp(:)'*A;
%     C2(:,k) = A'*temp(:);
%     C3(:,:,k) = A'*A;
%     C4(k) = norm(temp,'fro')^2;
% end
% B1 = sum(C1,3);
% B2 = sum(C2,2);
% B3 = sum(C3,3);
% B4 = sum(C4);
% 
% problem.cost = @(x) -B1*x - x'*B2 + trace(B3*x*x') + B4;
% problem.egrad = @(x) -2*B2 + 2*B3*x;
% 
% % checkgradient(problem);
% warning('off', 'manopt:getHessian:approx');
% options.verbosity = 0;
% [x,cost,info,options] = conjugategradient(problem, FRF(:), options);
% % [x,cost,info,options] = trustregions(problem, FRF(:));
% y = reshape(x,Nt,NRF);
% 
% end