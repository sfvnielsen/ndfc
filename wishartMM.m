function [z, E_Sigma, LL, par, noc_iter]=wishartMM(X,n,opts)
% Input
% X        array of covariance/scatter matrices (p x p x number of matrices)
% n        vector of the number of observations for which each covariance matrix is based
%          on
% opts.    struct with fields:
%
%
% Ouput
% z        assignment of covariances to clusters
% E_Sigma  expected value of covariance matrices
% LL       log of joint distribution value pr. iteration
% par      struct with model parameters and samples of z
% noc_iter number of components at each iteration
%
% Written by: Morten Mørup, mmor@dtu.dk
% Modified by: Søren Føns Vind Nielsen, sfvn@dtu.dk (Nov. 2016)
% This code is part of the Non-paramteric Dynamic Functional Connectivity
% Software (https://github.com/sfvnDTU/ndfc).
global tol_debug
tol_debug = 1e-8; % tolerance on log-joint test peforming in debug-mode
if ndims(X)>2
    [p,~,N]=size(X);
else
    [p,N]=size(X);
end

if isfield(opts,'Kinit'); Kinit=opts.Kinit; else Kinit=ceil(log(N)); end
if isfield(opts,'z'); z=opts.z; else z=ceil(Kinit*rand(N,1)); end
if isfield(opts,'eta'); eta=opts.eta; else eta=p; end
Sigma0 = eta*eye(p);
if isfield(opts,'v0'); v0=opts.v0; else v0=p; end
if isfield(opts,'maxiter'); maxiter=opts.maxiter; else maxiter=100; end; 
if isfield(opts,'alpha'); alpha=opts.alpha; else alpha=log(N); end; 
if isfield(opts,'debug'); debug=opts.debug; else debug=false; end; 

R=chol(Sigma0);
logPrior=v0*sum(log(diag(R)))-v0*p/2*log(2)-mvgammaln(p,v0/2);
val=unique(z);
Sigma_avg=zeros(p,p,length(val));
n_avg=zeros(1,length(val));
sumZ=zeros(1,length(val));
logP=zeros(1,length(val));
for k=1:length(val)
    idx=(z==val(k));
    z(idx)=k;
    if ndims(X)>2
        Sigma_avg(:,:,k)=sum(X(:,:,idx) ,3);
    else
        Sigma_avg(:,:,k)=X(:,idx)*X(:,idx)';
    end
    n_avg(k)=sum(n(idx));
    sumZ(k)=sum(idx);
    R=chol(Sigma_avg(:,:,k)+Sigma0);
    logdet(k)=2*sum(log(diag(R)));
    nn=(n_avg(k)+v0);
    logP(k)=logPrior-(nn/2*logdet(k)-nn*p/2*log(2)-mvgammaln(p,nn/2));
end
LL=nan(1,maxiter);
noc_iter=nan(1,maxiter);
par.eta = eta;
par.n_avg=n_avg;
par.sumZ=sumZ;
par.Sigma_avg=Sigma_avg;
par.Sigma0=Sigma0;
par.logPrior=logPrior;
par.v0=v0;
par.p=p;   
par.N=N;
par.alpha=alpha;
par.debug = debug;
LLbest=-inf;
ss=0;

%% Main loop
disp([' '])
disp('Infinite Wishart Mixture Model')
disp([' '])
disp(['To stop algorithm press control C'])
disp([' ']);
for iter=1:maxiter
    iter_start = tic;
    % Gibbs sample
    [z,par,logP]=gibbs_sample(X,n,z,logP,par,randperm(N));        
    
    % split-merge sample
    for k=1:max(z)
        [z,par,logP]=split_merge_sample(X,n,z,logP,par);    
    end    
    
    % sample alpha
    [logZ,par.alpha]=sample_alpha(par,par.alpha);        
    
    noc_iter(iter)=max(z);   
    LL(iter)=sum(logP)+logZ;%+const;    
    
  if LL(iter)>LLbest % Store best sample
        LLbest=LL(iter);
        E_Sigma=zeros(size(par.Sigma_avg));
        for k=1:size(par.Sigma_avg,3)
            E_Sigma(:,:,k)=(par.Sigma_avg(:,:,k)+par.Sigma0)/(par.n_avg(k)+v0-p-1);
        end
        par.sampleBest.E_Sigma=E_Sigma;
        par.sampleBest.z=z;
        par.sampleBest.par=par;        
        par.sampleBest.iter=iter;
    end
    if mod(iter,25)==0 % Store also every 25th sample
        ss=ss+1;
        par.sample(ss).z=z;
        par.sample(ss).Sigma0=par.Sigma0;
        par.sample(ss).alpha=par.alpha;
        par.sample(ss).iter=iter;
    end
        
    K=max(z);
    
    if mod(iter,10)==0|| iter==1
        fprintf('%12s | %15s | %12s | %12s | \n','Iteration','Log-Likelihood','# of states','Time [s]');
    end
    fprintf('%12d | %15d | %12d | %12.4f | \n',iter,LL(iter),K,toc(iter_start));

end
E_Sigma=zeros(size(par.Sigma_avg));
for k=1:size(par.Sigma_avg,3)
    E_Sigma(:,:,k)=(par.Sigma_avg(:,:,k)+par.Sigma0)/(par.n_avg(k)+v0-p-1);
end
end

%--------------------------------------------------------------------

% SUBFUNCTIONS

%--------------------------------------------------------------------
function [logZ,alpha]=sample_alpha(par,alpha)

max_iter=100;
K=length(par.sumZ);
N=par.N;
const=sum(gammaln(par.sumZ));
logZ=K*log(alpha)+const-gammaln(N+alpha)+gammaln(alpha);
accept=0;
for sample_iter=1:max_iter  
    alpha_new=exp(log(alpha)+0.1*randn);     % symmetric Proposal distribution in log-domain (use change of variable in acceptance rate alpha_new/alpha) 
    logZ_new=K*log(alpha_new)+const-gammaln(N+alpha_new)+gammaln(alpha_new); 
    if rand<(alpha_new/alpha*exp(logZ_new-logZ))
        alpha=alpha_new;
        logZ=logZ_new;
        accept=accept+1;       
    end
end    
end

%--------------------------------------------------------------------
function [z,par,logP]=split_merge_sample(X,n,z,logP,par)    
global tol_debug
  
i1=ceil(par.N*rand);
i2=ceil(par.N*rand);
while i2==i1
    i2=ceil(par.N*rand);
end
    
if z(i1)==z(i2) % Split move    
    % generate split configuration    
    z_t=z;    
    comp=[z(i1) max(z)+1];
    idx=(z_t==z(i1));
    z_t(idx)=comp(ceil(2*rand(sum(idx),1)));
    z_t(i1)=comp(1);
    z_t(i2)=comp(2);    
    idx(i1)=false;
    idx(i2)=false;
    logP_t=logP;
    par_t=par;
    if ndims(X)>2
        par_t.Sigma_avg(:,:,comp(1))=sum(X(:,:,z_t==comp(1)),3);
        par_t.Sigma_avg(:,:,comp(2))=sum(X(:,:,z_t==comp(2)),3);
    else
        par_t.Sigma_avg(:,:,comp(1))=X(:,z_t==comp(1))*X(:,z_t==comp(1))';
        par_t.Sigma_avg(:,:,comp(2))=X(:,z_t==comp(2))*X(:,z_t==comp(2))';
    end
    par_t.n_avg(comp(1))=sum(n(z_t==comp(1)));
    par_t.n_avg(comp(2))=sum(n(z_t==comp(2)));    
    R=chol(par_t.Sigma_avg(:,:,comp(1))+par.Sigma0);
    logdet(comp(1))=2*sum(log(diag(R)));           
    nn=(par_t.n_avg(comp(1))+par.v0);
    logP_t(comp(1))=par.logPrior-(nn/2*logdet(comp(1))-nn*par.p/2*log(2)-mvgammaln(par.p,nn/2));            
    R=chol(par_t.Sigma_avg(:,:,comp(2))+par.Sigma0);
    logdet(comp(2))=2*sum(log(diag(R)));           
    nn=(par_t.n_avg(comp(2))+par.v0);
    logP_t(comp(2))=par.logPrior-(nn/2*logdet(comp(2))-nn*par.p/2*log(2)-mvgammaln(par.p,nn/2));                                
    par_t.sumZ=par.sumZ;    
    par_t.sumZ(comp(1))=sum(z_t==comp(1));
    par_t.sumZ(comp(2))=sum(z_t==comp(2));    
    if sum(idx)>0
        for t=1:3
            [z_t,par_t,logP_t,logQ]=gibbs_sample(X,n,z_t,logP_t,par_t,find(idx)',comp);
        end
    else
        logQ=0;
    end
    
    logZ_t=max(z_t)*log(par.alpha)+sum(gammaln(par_t.sumZ))-gammaln(par.N+par.alpha)+gammaln(par.alpha);        
    logZ=max(z)*log(par.alpha)+sum(gammaln(par.sumZ))-gammaln(par.N+par.alpha)+gammaln(par.alpha);    
    %%% DEBUG %%%  LOG JOINT TEST
    if par.debug
        joint_new=evalLogJoint(X,n,z_t,par);
        joint_old=evalLogJoint(X,n,z,par);
        if abs( sum(logP_t)-sum(logP) - (joint_new - joint_old))/max(abs(joint_new),abs(joint_old)) > tol_debug    
            error('MyError:LogJointTestFailed', 'Log Joint Test failed in Split-Merge Sampler (SPLIT-MOVE)...')
        end
    end
    %%% DEBUG END %%%
    
    if rand<exp(sum(logP_t)+logZ_t-sum(logP)-logZ-logQ);
        disp(['split component ' num2str(z(i1))]);
        par=par_t;
        z=z_t;
        logP=logP_t;
    end                              
else % merge move
    % generate merge configuration
    Sigma_avg_new=par.Sigma_avg;    
    if ndims(X)>2
        Sigma_avg_new(:,:,z(i1))=par.Sigma_avg(:,:,z(i1))+sum(X(:,:,z==z(i2)),3);
    else
        Sigma_avg_new(:,:,z(i1))=par.Sigma_avg(:,:,z(i1))+X(:,z==z(i2))*X(:,z==z(i2))';
    end
    Sigma_avg_new(:,:,z(i2))=[];
    n_avg_new=par.n_avg;
    n_avg_new(z(i1))=par.n_avg(z(i1))+par.n_avg(z(i2));
    n_avg_new(z(i2))=[];
    sumZ_new=par.sumZ;
    sumZ_new(z(i1))=sumZ_new(z(i1))+sumZ_new(z(i2));
    sumZ_new(z(i2))=[];    
    
    % evaluate merge configuration
    z_new=z;
    comp=[z(i1) z(i2)];
    idx=(z==z(i1) | z==z(i2));
    z_new(idx)=z(i1);
    z_new(z_new>z(i2))=z_new(z_new>z(i2))-1;
    K_new=max(z_new);
    R=chol(Sigma_avg_new(:,:,z_new(i1))+par.Sigma0);
    logdet(z_new(i1))=2*sum(log(diag(R)));       
    logP_new=logP;    
    logP_new(z(i2))=[];
    nn=(n_avg_new(z_new(i1))+par.v0);
    logP_new(z_new(i1))=par.logPrior-(nn/2*logdet(z_new(i1))-nn*par.p/2*log(2)-mvgammaln(par.p,nn/2));            
    logZ_new=K_new*log(par.alpha)+sum(gammaln(sumZ_new))-gammaln(par.N+par.alpha)+gammaln(par.alpha);
    
    logZ=max(z)*log(par.alpha)+sum(gammaln(par.sumZ))-gammaln(par.N+par.alpha)+gammaln(par.alpha);
    
    accept_rate=rand;
    
    if accept_rate<exp(sum(logP_new)+logZ_new-sum(logP)-logZ);
        % generate split configuration    
        z_t=z;
        z_t(idx)=comp(ceil(2*rand(sum(idx),1)));
        z_t(i1)=z(i1);
        z_t(i2)=z(i2);    
        idx(i1)=false;
        idx(i2)=false;
        logP_t=logP;
        par_t=par;
        if ndims(X)>2
            par_t.Sigma_avg(:,:,z(i1))=sum(X(:,:,z_t==z(i1)),3);
            par_t.Sigma_avg(:,:,z(i2))=sum(X(:,:,z_t==z(i2)),3);
        else
            par_t.Sigma_avg(:,:,z(i1))=X(:,z_t==z(i1))*X(:,z_t==z(i1))';
            par_t.Sigma_avg(:,:,z(i2))=X(:,z_t==z(i2))*X(:,z_t==z(i2))';
        end
        par_t.n_avg(z(i1))=sum(n(z_t==z(i1)));
        par_t.n_avg(z(i2))=sum(n(z_t==z(i2)));    
        R=chol(par_t.Sigma_avg(:,:,z(i1))+par.Sigma0);
        logdet(z(i1))=2*sum(log(diag(R)));           
        nn=(par_t.n_avg(z(i1))+par.v0);
        logP_t(z(i1))=par.logPrior-(nn/2*logdet(z(i1))-nn*par.p/2*log(2)-mvgammaln(par.p,nn/2));            
        R=chol(par_t.Sigma_avg(:,:,z(i2))+par.Sigma0);
        logdet(z(i2))=2*sum(log(diag(R)));           
        nn=(par_t.n_avg(z(i2))+par.v0);
        logP_t(z(i2))=par.logPrior-(nn/2*logdet(z(i2))-nn*par.p/2*log(2)-mvgammaln(par.p,nn/2));                            

        par_t.sumZ=par.sumZ;    
        par_t.sumZ(z(i1))=sum(z_t==z(i1));
        par_t.sumZ(z(i2))=sum(z_t==z(i2));    
        if sum(idx)>0
            for t=1:2
                [z_t,par_t,logP_t]=gibbs_sample(X,n,z_t,logP_t,par_t,find(idx)',comp);
            end
            [z_t,par_t,logP_t,logQ]=gibbs_sample(X,n,z_t,logP_t,par_t,find(idx)',comp,z);        
        else
           logQ=0; 
        end
        
        %%% DEBUG %%%  LOG JOINT TEST
        if par.debug
            joint_new=evalLogJoint(X,n,z_new,par);
            joint_old=evalLogJoint(X,n,z,par);
            
            if abs( sum(logP_new)-sum(logP) - (joint_new - joint_old))/max(abs(joint_new),abs(joint_old)) > tol_debug
                error('MyError:LogJointTestFailed', 'Log Joint Test failed in Split-Merge Sampler (MERGE-MOVE)...')
            end
        end
        %%% DEBUG END %%%

        if accept_rate<exp(sum(logP_new)+logZ_new-sum(logP)-logZ+logQ);
            disp(['merged component ' num2str(z(i1)) ' with component ' num2str(z(i2))]);
            par.sumZ=sumZ_new;
            par.Sigma_avg=Sigma_avg_new;
            par.n_avg=n_avg_new;
            z=z_new;
            logP=logP_new;
        end
    end
end

% remove empty clusters
idx_empty=find(par.sumZ==0);
if ~isempty(idx_empty)
    par.Sigma_avg(:,:,idx_empty)=[];
    par.n_avg(idx_empty)=[];
    logP(idx_empty)=[]; 
    par.sumZ(idx_empty)=[];
    z(z>idx_empty)=z(z>idx_empty)-1;
end
end
%----------------------------------------------------------------------------
function [z,par,logP,logQ]=gibbs_sample(X,n,z,logP,par,sample_idx,comp,forced)
    if nargin<8
        forced=[];
    end    
    if nargin<7
        comp=[];
    end
    logQ=0;
    K=max(z);
    n_avg=par.n_avg;
    sumZ=par.sumZ;
    Sigma_avg=par.Sigma_avg;
    Sigma0=par.Sigma0;
    R=zeros(size(Sigma_avg));
    R0=chol(Sigma0);
    for k=1:size(R,3)
        R(:,:,k)=chol(Sigma_avg(:,:,k)+Sigma0);
    end
    logPrior=par.logPrior;
    v0=par.v0;
    p=par.p;    
    alpha=par.alpha;

    % gibbs sample clusters
    for i=sample_idx
       % remove i'th covariance matrix from all variables
       n_avg(z(i))=n_avg(z(i))-n(i);
       sumZ(z(i))=sumZ(z(i))-1;
       if ~ismatrix(X)
           Xi=X(:,:,i);
           Sigma_avg(:,:,z(i))=Sigma_avg(:,:,z(i))-Xi;
           R(:,:,z(i))=chol(Sigma_avg(:,:,z(i))+Sigma0);           
       else
           Xt=X(:,i);
           Xi=Xt*Xt';               
           Sigma_avg(:,:,z(i))=Sigma_avg(:,:,z(i))-Xi;
           R(:,:,z(i))=cholupdate(R(:,:,z(i)),Xt,'-');
       end
       logP(z(i))=logPrior-((n_avg(z(i))+v0)*sum(log(diag(R(:,:,z(i)))))-(n_avg(z(i))+v0)*p/2*log(2)-mvgammaln(p,(n_avg(z(i))+v0)/2));
              
       % Evaluate the assignment of i'th covariance matrix to all clusters
       logdet=zeros(1,K+1);
       if ~isempty(comp)
           sample_comp=comp;
       else
           sample_comp=1:K+1;
       end       
       if ~ismatrix(X)
            T=Sigma0+Xi;
            if ~isempty(comp)                      
                Q(:,:,comp)=bsxfun(@plus,Sigma_avg(:,:,comp),T);
            else                      
                Q=bsxfun(@plus,Sigma_avg,T);
            end
       end
       Rt=zeros([size(R,1), size(R,2), size(R,3)]+[0 0 1]);
       for k=sample_comp           
           if k<=K
               if ~ismatrix(X)
                   Rt(:,:,k)=chol(Q(:,:,k));
               else
                   Rt(:,:,k)=cholupdate(R(:,:,k),Xt,'+');
               end
               logdet(k)=2*sum(log(diag(Rt(:,:,k))));
           else
               if ~ismatrix(X)
                   Rt(:,:,k)=chol(T);
               else
                   Rt(:,:,k)=cholupdate(R0,Xt,'+');
               end
               logdet(k)=2*sum(log(diag(Rt(:,:,k))));
           end
       end       
       if isempty(comp)
            logP(K+1)=0;
            nn=[(n_avg+n(i)+v0) n(i)+v0];
            logPnew=logPrior-(nn/2.*logdet-nn*p/2*log(2)-mvgammaln(p,nn/2));            
            logDif=logPnew-logP;
            if par.debug %%% DEBUG
               passed = logJointTest_Gibbs(X,n,i,z,logDif,par,comp);
            end          %%% DEBUG END
            PP=[sumZ alpha].*exp(logDif-max(logDif));                     
       else            
            nn=n_avg(comp)+n(i)+v0;
            logPnew=logPrior-(nn/2.*logdet(comp)-nn*p/2*log(2)-mvgammaln(p,nn/2));            
            logDif=logPnew-logP(comp);
            if par.debug %%% DEBUG
               passed = logJointTest_Gibbs(X,n,i,z,logDif,par,comp);
            end          %%% DEBUG END
            PP=sumZ(comp).*exp(logDif-max(logDif));                     
       end
       
       % sample from posterior
       if isempty(comp)
            z(i)=find(rand<cumsum(PP/sum(PP)),1,'first');     
            logP(z(i))=logPnew(z(i));                            
       else
           if ~isempty(forced)
                z(i)=forced(i);                                
           else
                z(i)=comp(find(rand<cumsum(PP/sum(PP)),1,'first'));                                                 
           end
           q_tmp=logDif-max(logDif)+log(sumZ(comp));
           q_tmp=q_tmp-log(sum(exp(q_tmp)));     
           logQ=logQ+q_tmp(z(i)==comp);                
           logP(z(i))=logPnew(comp==z(i));                            
       end
       
       % Update sufficient statistics       
       if z(i)>K 
           K=K+1;                  
           n_avg(z(i))=n(i);
           sumZ(z(i))=1;
           Sigma_avg(:,:,z(i))=Xi;           
       else
           sumZ(z(i))=sumZ(z(i))+1;
           n_avg(z(i))=n_avg(z(i))+n(i);
           Sigma_avg(:,:,z(i))=Sigma_avg(:,:,z(i))+Xi;
       end
       R(:,:,z(i))=Rt(:,:,z(i));
       
       % remove empty clusters
       idx_empty=find(sumZ==0);
       if ~isempty(idx_empty)
        Sigma_avg(:,:,idx_empty)=[];
        R(:,:,idx_empty)=[];
        n_avg(idx_empty)=[];
        logP(idx_empty)=[]; 
        sumZ(idx_empty)=[];
        z(z>idx_empty)=z(z>idx_empty)-1;
        K=K-1;
       end
    end    
   
    par.n_avg=n_avg;
    par.sumZ=sumZ;
    par.Sigma_avg=Sigma_avg;    
    
    
end

%--------------------------------------------------------------------
function logJoint = evalLogJoint(X,n,z,par)
p = par.p;
N = par.N;
R=chol(par.Sigma0);
logPrior=par.v0*sum(log(diag(R)))-par.v0*p/2*log(2)-mvgammaln(p,par.v0/2);
val=unique(z);
Sigma_avg=zeros(p,p,length(val));
n_avg=zeros(1,length(val));
sumZ=zeros(1,length(val));
logP=zeros(1,length(val));
for k=1:length(val)
    idx=(z==val(k));
    z(idx)=k;
    if ndims(X)>2
        Sigma_avg(:,:,k)=sum(X(:,:,idx) ,3);
    else
        Sigma_avg(:,:,k)=X(:,idx)*X(:,idx)';
    end
    n_avg(k)=sum(n(idx));
    sumZ(k)=sum(idx);
    R=chol(Sigma_avg(:,:,k)+par.Sigma0);
    logdet(k)=2*sum(log(diag(R)));
    nn=(n_avg(k)+par.v0);
    logP(k)=logPrior-(nn/2*logdet(k)-nn*p/2*log(2)-mvgammaln(p,nn/2));
end
alpha=par.alpha;
logJoint = sum(logP);%+length(val)*log(alpha)+sum(gammaln(sumZ))-gammaln(N+alpha)+gammaln(alpha);
end



%--------------------------------------------------------------------
function passed = logJointTest_Gibbs(X,n,i,z,logDif,par,comp)
% join-likelihood test
global tol_debug
par1 = par; par2 = par;

% Sample two random states
if isempty(comp)
    z_pos = [unique(z);max(z)+1];
else
    z_pos = comp;
end
assert(length(z_pos)>1);

i1=ceil(length(z_pos)*rand);
i2=ceil(length(z_pos)*rand);
while i2==i1
    i2=ceil(length(z_pos)*rand);
end
z1 = z; z2 = z;
z1(i) = z_pos(i1); z2(i) = z_pos(i2);

[logJointTest_1] = evalLogJoint(X,n,z1,par1);
[logJointTest_2] = evalLogJoint(X,n,z2,par2);
passed =abs(logDif(i1)-logDif(i2)-(logJointTest_1-logJointTest_2))/max(abs(logJointTest_1),abs(logJointTest_2)) < tol_debug;
if ~passed
    error('MyError:LogJointTestFailed', 'Log Joint Test failed in Gibbs Sampler...')
end
%eof
end


