function [H] = SVMIMOmodel(K,Nt,Nr,Ncl,Nray,std)
    d_lamda = 0.5;
    Ct = [0:Nt-1]';
    Cr = [0:Nr-1]';
    H = zeros(K,Nr,Nt);
    
    for k = 1:K
        Htemp = zeros(Nr,Nt);
        for ii = 1:Ncl
            phi_i = unifrnd(0,2*pi);
            the_i = unifrnd(0,2*pi);
            for jj = 1:Nray
                a = (randn()+1i*randn())/sqrt(2);
                phi_ij = spreadAoD(phi_i,std);
                ft_Nt = 1/sqrt(Nt)*exp(Ct*1i*2*pi*d_lamda*sin(phi_ij));
                ft_Nt = a*ft_Nt';%'
                b = (randn()+1i*randn())/sqrt(2);
                the_ij = spreadAoD(the_i,std);
                ft_Nr = 1/sqrt(Nr)*exp(Cr*1i*2*pi*d_lamda*sin(the_ij));
                ft_Nr = b*ft_Nr;
                Htemp = Htemp + ft_Nr*ft_Nt;
            end
        end
        H(k,:,:) = Htemp;
    end
    H = H*sqrt(Nt*Nr/Ncl/Nray);
end

function [x] = spreadAoD(mu,std)
    b=std/sqrt(2);
    a=rand()-0.5;
    x=mu-b*sign(a).*log(1-2*abs(a));
end

