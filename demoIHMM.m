%% Demo script for IHMM
p = 5;
T_SCALE = 500;
N=T_SCALE*[0.4 0.3 0.2 0.1];
ZS = [1,2,3,1];

X=zeros(p,sum(N));
X_test = zeros(p,sum(N));

% Generate state specific parameters
for n = 1:length(unique(ZS))
    R{n}=triu(randn(p));
    M{n}=4*randn(p,1);
end

% Generate data and independent test data
zt = [];
for n=1:length(N)
   X(:,sum(N(1:n-1))+1:sum(N(1:n)))=R{ZS(n)}'*randn(p,N(n))+repmat(M{ZS(n)},[1,N(n)]);   
   X_test(:,sum(N(1:n-1))+1:sum(N(1:n)))=R{ZS(n)}'*randn(p,N(n))+repmat(M{ZS(n)},[1,N(n)]);   
   zt = [zt n*ones(1,N(n))];
end


%% IHMM analysis
opts.Sigma0_scale = 1e-3*std(X(:)); % weak prior due to good SNR
opts.maxiter=250;
opts.emission_type = 'SSM';

[z, E, LL, par,samp]=IHMMgibbs(X,opts);


figure, subplot (1,2,1)
bar(zt)
title('True State Sequence')
subplot(1,2,2)
bar(z)
title('Estimated State Sequence')


%% Predictive likelihood on test data
predictiveLikelihood(X_test,samp)


pred
