% Arthur's ridge function.
% This function implements MATLAB's ridge regression with an additional feature of speeding up computations
% when the number of features outweigh the number of observations greatly. Usage is the same as
% MATLAB's default ridge function. The function checks the dimension of X to automatically decide
% whether to use MATLAB's algorithm, which is adding pseudo observations, or to use the new algorithm.
%
% The new algorithm is based on the equality that
% (X'*X + k*I)\X' = X'/(X*X' + k*I), which I learned from the website below
% https://danieltakeshi.github.io/2016/08/05/a-useful-matrix-inverse-equality-for-ridge-regression/
% for example, if X is a 50-by-1000 matrix
% X = rand(50,1000); k = 0.1;
% tic; out1 = (X'*X + k*eye(1000))\X'; toc
% tic; out2 = X'/(X*X' + k*eye(50)); toc
% the first one takes 0.016403 seconds for me while the second one takes 0.001312 seconds
% this is because X'*X is a very large matrix (1000 by 1000) while X*X' is quite small (50 by 50)
%
% Using this example, let
% X = rand(50,1000); Y = X*rand(1000,1);
% tic; b1 = ridge(Y,X,logspace(-3,0,10)); toc  % took 0.894749 seconds
% tic; b2 = ridge_fastwide(Y,X,logspace(-3,0,10)); toc  % took 0.010885 seconds
% norm(b1-b2)   % less than 1e-07, hence no difference in coefficients
%
% In normal situations when number of observations are greater, they use the same code, hence b should be exactly the same

function b = ridge_fastwide(y,X,k,flag)
    if nargin < 3
        error(message('stats:ridge:TooFewInputs'));
    end

    if nargin<4 || isempty(flag) || isequal(flag,1)
        unscale = false;
    elseif isequal(flag,0)
        unscale = true;
    else
        error(message('stats:ridge:BadScalingFlag'));
    end

    % Check that matrix (X) and left hand side (y) have compatible dimensions
    [n,p] = size(X);

    [n1,collhs] = size(y);
    if n~=n1
        error(message('stats:ridge:InputSizeMismatch'));
    end

    if collhs ~= 1
        error(message('stats:ridge:InvalidData'));
    end

    % Remove any missing values
    wasnan = (isnan(y) | any(isnan(X),2));
    if (any(wasnan))
        y(wasnan) = [];
        X(wasnan,:) = [];
        n = length(y);
    end

    % Normalize the columns of X to mean zero, and standard deviation one.
    mx = mean(X);
    stdx = std(X,0,1);
    idx = find(abs(stdx) < sqrt(eps(class(stdx))));
    if any(idx)
        stdx(idx) = 1;
    end

    MX = mx(ones(n,1),:);
    STDX = stdx(ones(n,1),:);
    Z = (X - MX) ./ STDX;
    if any(idx)
        Z(:,idx) = 1;
    end



    % decide which algorithm to use
    nk = numel(k);
    if p > n % more features than observations
        Covmat = Z*Z';
        Z = Z'; % pre-transposing to avoid transposing repeatedly

        % Compute the coefficient estimates
        b = Z/(Covmat+k(1).*eye(n))*y;

        if nk>1
            % Fill in more entries after first expanding b.  We did not pre-
            % allocate b because we want the backslash above to determine its class.
            b(end,nk) = 0;
            for j = 2:nk
                b(:,j) = Z/(Covmat+k(j).*eye(n))*y;
            end
        end
    else
        % Compute the ridge coefficient estimates using the technique of
        % adding pseudo observations having y=0 and X'X = k*I.
        pseudo = sqrt(k(1)) * eye(p);
        Zplus  = [Z;pseudo];
        yplus  = [y;zeros(p,1)];

        % Compute the coefficient estimates
        b = Zplus\yplus;

        if nk>1
            % Fill in more entries after first expanding b.  We did not pre-
            % allocate b because we want the backslash above to determine its class.
            b(end,nk) = 0;
            for j=2:nk
                Zplus(end-p+1:end,:) = sqrt(k(j)) * eye(p);
                b(:,j) = Zplus\yplus;
            end
        end
    end



    % Put on original scale if requested
    if unscale
        b = b ./ repmat(stdx',1,nk);
        b = [mean(y)-mx*b; b];
    end
end