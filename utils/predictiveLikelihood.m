function [avg_llike,llike] = predictiveLikelihood(X,post_samples)
% Calculates the predictive likelihood of data X (dimension p * T)
% Based on a modified Viterbi-algorithm
% INPUT:
%           X - (p x T) data matrix with T data points each of dimension p
%           post_samples - struct constaining posterior parameter samples
%                          from model. Contains the fields:
%                 A - VAR coefficients
%                 S - Noise covariance
%                 Pi - Transition matrix
%                 z - state sequence
%
%
% OUTPUT:
%           avg_llike - the avg log-likelihood over all samples
%           llike   - the log-likelihood output from one Viterbi pass from
%                     each sample
% Written by: Søren Føns Vind Nielsen, Technical University of Denmark, 2015


[p,T] = size(X);
n_samples = length(post_samples.S);
llike = nan(n_samples,1);
if isfield(post_samples,'A')
    M = ceil( size(post_samples.A{1},2)/p);
else
    M = 0;
end

if M>0
    % Extract past based on model order of each process (M)
    X_past = nan(p*M,T-M);
    for t = 1:M
        ids = (1+p*(t-1)):(1+p*(t-1))+p-1;
        X_past(ids,:) = X(:,(M-(t-1)):(T-t));
    end
    X(:,1:M) = [];
    T = T-M;
else % if model was not auto-regressive set past to an appropriate zero matrix
    X_past = zeros(p,T);
end

for n = 1:n_samples
    % Extract this samples parameters and state sequence
    z = post_samples.z{n};
    K = max(z);
    
    
    % Cholesky-factorize S for speed in later calculations
    cS = post_samples.S{n};
    for k = 1:size(cS,3)
        cS(:,:,k) = chol(cS(:,:,k));
    end
    
    Trans = post_samples.Pi{n};
    
    
    % If initial state distribution is sampled...
    if isfield(post_samples,'emp_state')
        emp_state_dist = post_samples.emp_state{n};
    else
        % Empirical state distribution (from sequence)
        emp_state_dist = hist(z,unique(z))/length(z);
    end
    
    % Extract emission mean
    if isfield(post_samples,'A')
        A = post_samples.A{n};
        MU = zeros(p,size(X,2),K);
        for k = 1:K
            MU(:,:,k) = A(:,:,k)*X_past;
        end
    elseif isfield(post_samples,'mu')
        mu = post_samples.mu{n};
        MU = zeros(p,size(X,2),K);
        for k = 1:K
            MU(:,:,k) = repmat(mu(:,k),[1,size(X,2)]);
        end
    else
        MU = zeros(p,size(X,2),K);
    end
    
    % Calculate all emissions
    emission = nan(K,T);
    for k = 1:K
        emission(k,:) = evalLoglLike(X,MU(:,:,k),cS(:,:,k));
    end
    
    % Initialize Viterbi chain V
    V = zeros(K,T);
    for k = 1:K
        V(k,1) = log(emp_state_dist(k)) + emission(k,1);
    end
    
    % Run through chain
    for t = 2:T
        for k = 1:max(z)
            % Sum previous Viterbi state times transition and multiply with
            % emission
            if K>1
                V(k,t) = lsum(V(:,t-1)+ log(Trans(:,k)),1) + emission(k,t);
            else
                V(k,t) = lsum(V(:,t-1),1) + emission(k,t);
            end
        end
    end
    % End of chain - sum together
    llike(n) = lsum(V(:,end),1);
end

% Return avg of all samples
avg_llike =  lsum(llike,1)-log(n_samples);
%eof
end

function ll = evalLoglLike(x,MU,cS)
p = size(x,1);
pX = (x-MU)'/cS;
ll =-p/2*log(2*pi)-sum(log(diag(cS)))-1/2*sum(pX.*pX,2);
end

function Y = lsum(X,d)
% LSUM Numerically stable computation of log(sum(exp(X),d))
% Written by: Mikkel N. Schmidt, CogSys, DTU
% www.mikkelschmidt.dk
maxX = max(min(max(X, [], d),realmax),-realmax);
Y = maxX + log(sum(exp(bsxfun(@minus, X, maxX)),d));
end