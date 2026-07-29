function [V] = MIMOWMMSE(H,Ns,SNR)
epsilon = 1e-5;
sigma2=10^(-SNR/10);
[K,Nr,Nt] = size(H);
Keq = K*Ns;
% mode = "rzf";
mode = "random";

if strcmp(mode, 'rzf') 
    n_iter = 1;
elseif strcmp(mode, 'random')
    n_iter = 5;
end

H_cell = cell(K, 1);
% convert channel tensor to channel cell
for k = 1:K
    H_cell{k} = reshape(H(k,:,:), [Nr, Nt]);
end

% for different initialization
V_iter_res = cell(1,n_iter);
rate_iter_res = zeros(1,n_iter);

for i_iter = 1: n_iter
    if i_iter == 1
        % for SVD calculation of channel
        Q = cell(K, 1);
        S = cell(K, 1);
        T = cell(K, 1);
        Hsvd = cell(K,1);
        
        % user-wise SVD
        for k = 1:K
            [Q{k}, S{k}, T{k}] = svd(H_cell{k},"econ");
            % virtual channel
            Hsvd{k} = S{k}(1:Ns,1:Ns)*T{k}(:,1:Ns)';
        end
        Hsvd_stack = vertcat(Hsvd{:});
        V_stack = Hsvd_stack'*((Hsvd_stack*Hsvd_stack'+Keq*sigma2*eye(Keq))\eye(Keq));
        V_stack = V_stack./sqrt(sum(abs(V_stack).^2))/sqrt(Keq);
        %
    else
        V_stack = 10.^(2*rand(1,Keq)-1).*(randn(Nt,Keq) + 1i*randn(Nt,Keq));
        V_stack = V_stack./sqrt(sum(abs(V_stack).^2))/sqrt(Keq);
    end

    % variables initialization(e.g.,V:svd+rzf, U&W:any)
    V = mat2cell(V_stack, Nt, Ns * ones(K, 1));
    U = cellfun(@(~) zeros(Nr, Ns), cell(K, 1), 'UniformOutput', false);
    W = cellfun(@(~) zeros(Ns, Ns), cell(K, 1), 'UniformOutput', false);
    Wre = cellfun(@(~) zeros(Ns, Ns)+100*eye(Ns), cell(K, 1), 'UniformOutput', false);

    Time = 100;%正常情况十几次就收敛
    while(abs(sum(log2(cellfun(@(x) det(x), W)))-sum(log2(cellfun(@(x) det(x), Wre))))>epsilon && Time>0)
        Time = Time - 1;
        Wre = W;
        for m = 1:K
            temp = zeros(Nr,Nr);
            for i = 1:K
                temp = temp + H_cell{m}*V{i}*V{i}'*H_cell{m}';
            end
            U{m} = (temp+sigma2*eye(Nr))\H_cell{m}*V{m};
            W{m} = eye(Ns)/(eye(Ns)-U{m}'*H_cell{m}*V{m});
        end

        %calculate mu with bisection methods
        S1 = 0;
        for k=1:K
               S1 = S1 + H_cell{k}'*U{k}*W{k}*U{k}'*H_cell{k};
        end
        [D,L] = eig(S1);
        S2 = 0;
        for k=1:K
               S2 = S2 + H_cell{k}'*U{k}*W{k}*W{k}'*U{k}'*H_cell{k};
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
           V{m} =(S1+mu*eye(Nt))\H_cell{m}'*U{m}*W{m};
        end
    end
    V_iter_res{i_iter} = V;
    rate_iter_res(i_iter) = MIMOcalrate(H,V,SNR);
end
 
[~,id] = max(rate_iter_res);
V = V_iter_res{id};
