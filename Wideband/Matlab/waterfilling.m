function p = waterfilling(h, Pt, sigma2)
    % Water-filling algorithm with total power constraint
    % Inputs:
    %   h      : complex channel vector (Nx1 or 1xN)
    %   sigma2 : noise power (scalar)
    % Output:
    %   p      : optimal power allocation (Nx1 vector)
    
    % Convert channel gains to real positive gains
    g = abs(h).^2;
    N = length(g);
    P_total = Pt;

    % Set precision and water level bounds
    tol = 1e-5;
    mu_low = sigma2 / max(g);          % lower bound of water level
    mu_high = sigma2 / min(g) + P_total;  % upper bound of water level

    % Bisection method to find water level
    while mu_high - mu_low > tol
        mu = (mu_high + mu_low) / 2;
        p_temp = max(mu - sigma2 ./ g, 0);
        if sum(p_temp) > P_total
            mu_high = mu;
        else
            mu_low = mu;
        end
    end

    % Final power allocation
    p = max(mu - sigma2 ./ g, 0);
end
