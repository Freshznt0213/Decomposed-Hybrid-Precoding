function [V] = WMMSE(H,SNR)
epsilon = 1e-7;
sigma2=10^(-SNR/10);
[K,N] = size(H);
mode = "rzf";
% mode = "random";

if strcmp(mode, 'rzf') 
    n_iter = 1;
elseif strcmp(mode, 'random')
    n_iter = 5;
end

V_cell = cell(1,n_iter);
rate = zeros(1,n_iter);

for i_iter = 1: n_iter
    if i_iter == 1
        %用RZF初始化V
        V = H'*((H*H'+K*sigma2*eye(K))\eye(K));
        V = V./sqrt(sum(abs(V).^2))/sqrt(K);
        %
    else
        V = 10.^(2*rand(1,K)-1).*(randn(N,K) + 1i*randn(N,K));
        V = V./sqrt(sum(abs(V).^2))/sqrt(K);
    end
    U = zeros(1,K);
    W = zeros(1,K);
    Wre = W+100;
    Time = 100;%正常情况十几次就收敛
    while(abs(sum(log2(abs(W)))-sum(log2(abs(Wre))))>epsilon && Time>0)
        Time = Time - 1;
        Wre = W;
        % calculate U and W
        for m = 1:K
            temp = 0;
            for i = 1:K
                temp = temp + H(m,:)*V(:,i)*V(:,i)'*H(m,:)';
            end
            U(m) = 1/(temp+sigma2)*H(m,:)*V(:,m);
            W(m) = 1/(1-U(m)'*H(m,:)*V(:,m));
        end

        %算mu，只有1个正实数解
        S = 0;
        for k=1:K
               S = S + H(k,:)'*U(k)*W(k)*U(k)'*H(k,:);
        end
        [D,L] = eig(S);
        S2 = 0;
        for k=1:K
               S2 = S2 + H(k,:)'*U(k)*W(k)*W(k)'*U(k)'*H(k,:);
        end
        P = D'*S2*D;
        P = diag(abs(P));
        L = diag(abs(L));

        a = 0;%% 求mu s.t.sum(P./((L+mu).^2))-1==0
        b = 100;%%note
        ind = b-a;
        while ind > 1e-5
            xx = (a+b)/2;
            if (sum(P./((L+a).^2))-1)*(sum(P./((L+xx).^2))-1) < 0
                b = xx;
            else
                a = xx;
            end
            ind = b - a;
        end
        mu = xx;

        %新Vm
        for m=1:K
           V(:,m) =(S+mu*eye(N))\H(m,:)'*U(m)*W(m);
        end
    end
    V_cell{i_iter} = V;
    rate(i_iter) = calrate(H,V,K,SNR);
end
 
[~,id] = max(rate);
V = V_cell{id};
