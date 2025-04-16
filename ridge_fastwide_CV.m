function [b, perfmat,bestk] = ridge_fastwide_CV(y,X,k,CVfold,perfmetric)
    % perform cross-validation
    [CVfold,nfold] = prepCVfold(CVfold);

    perfmat = nan(nfold,length(k));
    for i = 1:nfold
        b = ridge_fastwide(y(CVfold(:,i)==0),X(CVfold(:,i)==0,:),k,0);
        score = b(1,:)+X(CVfold(:,i)==1,:)*b(2:end,:);
        switch perfmetric
            case 'Pearson'
                perfmat(i,:) = corr(y(CVfold(:,i)==1),score);
            case 'negMSE'
                perfmat(i,:) = -mean((y(CVfold(:,i)==1)-score).^2,1);
            otherwise
                error('unknown perfmetric')
        end
    end
    [~,ind] = max(mean(perfmat));
    bestk = k(ind);

    % final model
    b = ridge_fastwide(y,X,bestk,0);
end

% prepare CV fold data into a matrix form, which is more generalizable
function [CVfold,nfold] = prepCVfold(inCVfold)
    if size(inCVfold,2) == 1 % vector
        uniqfold = unique(inCVfold); nfold = length(uniqfold);
        CVfold = zeros(length(inCVfold),nfold);
        for i = 1:nfold
            CVfold(:,i) = 1.*(inCVfold == uniqfold(i));
        end
    elseif size(inCVfold,2) > 1 % matrix
        nfold = size(inCVfold,2); CVfold = inCVfold;
        if any(CVfold(:)~=0 & CVfold(:)~=1)
            error('Non-binary element in matrix form CVfold. Perhaps you meant to use vector form?')
        end
    else
        error('unexpected size of CVfold')
    end
end