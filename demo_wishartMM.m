%% Demo script for IWMM
addpath utils/

p = 10;
T_SCALE = 500;
N=T_SCALE*[0.4 0.3 0.2 0.1];
ZS = [1,2,3,1];

X=zeros(p,sum(N));
X_test = zeros(p,sum(N));

% Generate state specific parameters
for n = 1:length(unique(ZS))
    R{n}=triu(randn(p));
end

% Generate data and independent test data
zt = [];
for n=1:length(N)
   X(:,sum(N(1:n-1))+1:sum(N(1:n)))=R{ZS(n)}'*randn(p,N(n));   
   X_test(:,sum(N(1:n-1))+1:sum(N(1:n)))=R{ZS(n)}'*randn(p,N(n));   
   zt = [zt ZS(n)*ones(1,N(n))];
end

% Extract scatter matrices
wl = 25;
nwin = size(X,2)/wl;
XX = nan(p,p,nwin); XX_test = nan(p,p,nwin);
df = nan(1,nwin); zwin = nan(1,nwin);
for ii = 1:nwin
    XX(:,:,ii) = X(:, (1:wl)+(ii-1)*wl)*X(:, (1:wl)+(ii-1)*wl)' ;
    XX_test(:,:,ii) = X_test(:, (1:wl)+(ii-1)*wl)*X_test(:, (1:wl)+(ii-1)*wl)';
    df(ii) = wl; % degrees of freedom for specifc window
    zwin(ii) = zt( 1 + (ii-1)*wl );
end


%% Run inference
opts = struct();
opts.eta = 1e-3*std(X(:)); % weak prior due to good SNR - should be cross-validated in practice
[z_wishmm, E_Sigma, LL, par, noc_iter]=wishartMM(XX,df,opts);


%% Results
figure, subplot (1,2,1)
bar(zwin)
title('True State Sequence')
axis tight
subplot(1,2,2)
bar(z_wishmm)
title('Estimated State Sequence')
axis tight

% Predictive Likelihood 
predictive_loglike =sum(wishartMM_predict(XX_test,df,XX,df,par,par.sample))
% NB! Currently only supports calculating full predictive likelihood - i.e.
% all slices XX_test should positive-definite