function R = calrate(H,W,K,sigma2)
%H[K*Ns,Nt,NRB],W:[Nt,K*Ns,NRB]
Q = abs(pagemtimes(H,W)).^2;
D = eye(K).* Q;
sinr_set = sum(D,2)./(sigma2+sum(Q,2)-sum(D,2));
R = sum(log2(1+sinr_set));
R = mean(R);
end