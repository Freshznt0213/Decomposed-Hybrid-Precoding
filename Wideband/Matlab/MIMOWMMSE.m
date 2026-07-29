function [V_cell] = MIMOWMMSE(H_cell,V_rzf,Ns,Pt,sigma2)
% H :[NRB, K,Nr,Nt]
epsilon = 1e-7;
[NRB, K] = size(H_cell);
[Nr, Nt] = size(H_cell{1});
Keq = K*Ns;
mode = "rzf";
% mode = "random";

if strcmp(mode, 'rzf') 
    n_iter = 1;
elseif strcmp(mode, 'random')
    n_iter = 5;
end
V_cell = cell(NRB,K);
SE = 0;


%for different RBs
for nrb = 1:NRB
    H_cell_nrb = H_cell(nrb,:);
    % for different initialization
    V_iter_res = cell(1,n_iter);
    rate_iter_res = zeros(1,n_iter);
    for i_iter = 1: n_iter
        if i_iter == 1
            V = V_rzf(nrb,:);
        else
            V_stack = 10.^(2*rand(1,Keq)-1).*(randn(Nt,Keq) + 1i*randn(Nt,Keq));
            V_stack = V_stack./sqrt(sum(abs(V_stack).^2))/sqrt(Keq)*sqrt(Pt);
            V = mat2cell(V_stack, Nt, Ns * ones(K, 1));
        end
    
        % variables initialization(e.g.,V:svd+rzf, U&W:any)
        % V = mat2cell(V_stack, Nt, Ns * ones(K, 1));

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
                    temp = temp + H_cell_nrb{m}*V{i}*V{i}'*H_cell_nrb{m}';
                end
                U{m} = (temp+sigma2*eye(Nr))\H_cell_nrb{m}*V{m};
                W{m} = eye(Ns)/(eye(Ns)-U{m}'*H_cell_nrb{m}*V{m});
            end
    
            %calculate mu with bisection methods
            S1 = 0;
            for k=1:K
                   S1 = S1 + H_cell_nrb{k}'*U{k}*W{k}*U{k}'*H_cell_nrb{k};
            end
            [D,L] = eig(S1);
            S2 = 0;
            for k=1:K
                   S2 = S2 + H_cell_nrb{k}'*U{k}*W{k}*W{k}'*U{k}'*H_cell_nrb{k};
            end
            P = D'*S2*D;
            P = diag(abs(P));
            L = diag(abs(L));
    
            a = 0;%% 求mu s.t.sum(P./((L+mu).^2))-1==0
            b = 100;%%note
            ind = b-a;
            while ind > 1e-5
                xx = (a+b)/2;
                if (sum(P./((L+a).^2))-Pt)*(sum(P./((L+xx).^2))-Pt) < 0
                    b = xx;
                else
                    a = xx;
                end
                ind = b - a;
            end
            mu = xx;
    
            %新Vm
            for m=1:K
               V{m} =(S1+mu*eye(Nt))\H_cell_nrb{m}'*U{m}*W{m};
            end
        end
        V_iter_res{i_iter} = V;
        rate_iter_res(i_iter) = MIMOcalrate(H_cell_nrb,V,sigma2);
    end
    [~,id] = max(rate_iter_res);
    V = V_iter_res{id};
    V_cell(nrb,:) = V;
end