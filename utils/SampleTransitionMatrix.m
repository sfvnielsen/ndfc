function [ Pi ] = SampleTransitionMatrix( S, H , sticky)
%SAMPLETRANSITIONMATRIX Samples a transition matrix from a state sequence S
% and Dirichlet prior H (row vector).
if nargin<3
    sticky = 0;
end

K = size(H,2);
T = length(S);
Pi = zeros(K);

N = zeros(K);
for t=2:T
    N(S(t-1), S(t)) = N(S(t-1), S(t)) + 1;
end

for k=1:K
    kappa = zeros(1,K); kappa(k) = sticky;
    Pi(k, :) = dirichlet_sample(N(k,:) + H + kappa );
end