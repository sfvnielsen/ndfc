function [z_mj, E, LL, par_mj,post_samples]=IHMMgibbs(X,opts,sampling_state)
% INFINITE HIDDEN MARKOV MODEL GIBBS SAMPLER WITH SPLIT-MERGE MOVES
%
% Function for Infinte Hidden Markov Model (IHMM) with time specific
% scaling parameter and stickyness. 
% Emission model can be on of the three following: 
%       1) Zero mean Gaussian (State specific covariance) (ZMG)
%       2) State specific mean and covariance (SSM)
%       3) State specific auto-regressive mean and state-covariance (VAR)
% Inference is done using Gibbs sampling based on:
% Beal, Matthew J., Zoubin Ghahramani, and Carl E. Rasmussen.
% "The infinite hidden Markov model."
% Furthermore a
% Advances in neural information processing systems. 2001.
% Implementation is based on Juergen Van Gaels IHMM-Toolbox:
% http://mloss.org/software/view/205/
%
%
% Example of function call (with default settings - cf. below):
%   [z,E_Sigma,LL,par,post_samples] = IHMMgibbs(X,struct());
%
% Input:
% X        p x N matrix of regions/voxels x time
%           NB! For multiple subjects - concatenate in time and use
%           begin_i option.
%
% opts.    struct with fieds:
%           emission_type   type of emission model
%                                   'ZMG' = Zero Mean Gaussian
%                                   'SSM' = State Specific Mean
%                                   'VAR' = Vector Autoregressive Model
%           Kinit           initial number of clusters (default: log(N))
%           z               initial clustering configuration
%           Sigma0          initial prior on covariances (default: eye(p))
%           v0              degrees of freedom parameter (covariance), this parameter is treated as fixed (default: p)
%           maxiter         maximum number of sampling iterations (default: 100)
%           alpha           prior parameter for IHMM (default: 1)
%           gamma           prior parameter for IHMM (default: 1)
%           beta            prior parameter for IHMM (default: ones(1,Kinit+1)/(Kinit+1))
%           kappa           prior parameter for IHMM - controls stickyness
%                           (default = 0 = non-sticky)
%           begin_i         1 x N boolean vector with true where first sample occur in the chain (default: begin_i(1)=true; begin_i(2:end)=false; )
%           const_kappa     constant kappa (stickyness) parameter
%                           (default: true)
%           split_merge     use split-merge moves (default: true)
%           const_state     force all data points to one state  (default: false)
%           cpoint_factor   number of times change points should be sampled
%                           addtionally (default: 0)
%           post_thinning   posterior sampling thinning -
%                            the number of iterations between sampling parameter posteriors
%                           (default: 10)
%           burnin          number of iterations before starting collecting
%                           posterior samples (default: maxiter/2)
%           max_annealing   run annealing - i.e. attempt converging to the
%                           optimum (default: false)
%
%       debug               enable debug-mode - checks joint likelihood vs.
%                           conditional in some of the samplers
%       debug_print         verbosing debug mode
%       keyboard_interrupt  allows for keyboard-mode at different places in the code
%       debug_save          saves state of algorithm in debug mode if error occurs
%
%
% Ouput
% z_mj        assignment data points to processes (max joint probability)
% E           struct - with mean estimates of parameters of interest
%                       Sigma - Covariances
% LL           log of joint distribution value pr. iteration
% par_mj      parameter struct from MAP solution
% post_samples parameter posterior samples (wrapped in struct)
%
% Feel free to use the code but please throw a reference of some kind :-)
% This code is provided as is without any kind of warranty or liability!
%
%
% Written by: Morten Mørup and Søren Føns Vind Nielsen
% Technical University of Denmark, Spring 2016
%%%%
global Xtrue Xpast
Xtrue = X;
[p,N]=size(X);

if nargin==3
    opts = sampling_state.opts;
end
if isfield(opts,'emission_type'); par.emission_type=opts.emission_type; else par.emission_type='SSM'; end
if isfield(opts,'begin_i'); begin_i=opts.begin_i; else begin_i=false(1,N); begin_i(1)=true; end;
if isfield(opts,'Kinit'); Kinit=opts.Kinit; else Kinit=ceil(log(N)); end
switch par.emission_type
    case 'VAR'
        if isfield(opts,'M'); M=opts.M; else M=1; end
        if isfield(opts,'tau_M'); tau_M=opts.tau_M; else tau_M=ones(M,1); end
        task_id_begin = find(begin_i);
        task_id_end = task_id_begin-1; task_id_end = [task_id_end(2:end),N];
        n_tasks = length(task_id_begin);
        if isfield(opts,'const_state'); const_state = opts.const_state; z = ones(N-n_tasks*M,1); else const_state=false;end;
        if isfield(opts,'z'), z=opts.z; elseif const_state, else z=ceil(Kinit*rand(N-n_tasks*M,1)); end
    case {'ZMG','SSM'}
        if isfield(opts,'const_state'); const_state = opts.const_state; z = ones(N,1); else const_state=false;end;
        if isfield(opts,'z'), z=opts.z; elseif const_state, else z=ceil(Kinit*rand(N,1)); end
    otherwise
        error('Unrecognized Emisssion Type: Choose between ZMG, SSM and VAR')
end

if isfield(opts,'Sigma0'); Sigma0=opts.Sigma0; else Sigma0=eye(p); end
if isfield(opts,'mu0'); mu0=opts.mu0; else mu0=zeros(p,1); end
if isfield(opts,'lambda'); lambda = opts.lambda; else lambda=std(X(:)); end
if isfield(opts,'Sigma0_scale'); Sigma0_scale=opts.Sigma0_scale; else Sigma0_scale=1; end
if isfield(opts,'v0'); v0=opts.v0; else v0=p; end
if isfield(opts,'maxiter'); maxiter=opts.maxiter; else maxiter=100; end;
if isfield(opts,'alpha'); alpha=opts.alpha; else alpha=1; end;
if isfield(opts,'gamma'); gamma=opts.gamma; else gamma=1; end;
if isfield(opts,'beta'); beta=opts.beta; else beta=ones(1,Kinit+1)/(Kinit+1); end;
if isfield(opts,'kappa'); kappa = opts.kappa; else kappa=0; end

if isfield(opts,'const_kappa'); const_kappa=opts.const_kappa; else const_kappa=true;end;
if isfield(opts,'split_merge'); split_merge=opts.split_merge; else split_merge=true;end;
if isfield(opts,'cpoint_factor'); cpoint_factor = opts.cpoint_factor; else cpoint_factor=0;end;
if isfield(opts,'max_annealing'); max_annealing = opts.max_annealing; else max_annealing=false;end;
if isfield(opts,'burnin'); burnin = opts.burnin; else burnin=ceil(maxiter/2);end;
if isfield(opts,'post_thinning'); post_thinning = opts.post_thinning; else post_thinning=10;end;

if isfield(opts,'debug'); debug=opts.debug; else debug=false;end;
if isfield(opts,'debug_print'); debug_print=opts.debug_print; else debug_print=false;end;
if isfield(opts,'keyboard_interrupt'); keyboard_interrupt=opts.keyboard_interrupt; else keyboard_interrupt=false;end;
if isfield(opts,'save_every'); save_every=opts.save_every; else save_every=0;end;
if isfield(opts,'save_file'); save_file=opts.save_file; else save_file='ihmmgibbs_results.mat';end;
if isfield(opts,'verbose'); par.verbose=opts.verbose; else par.verbose=false;end;

%% Initialization
% hyperparametesr set to values used in iHMM-v0.5__ by Jurgen Van Gael
hypers.alpha0_a = 4;
hypers.alpha0_b = 2;
hypers.gamma_a = 3;
hypers.gamma_b = 6;

if nargin<3
post_samples.S = cell(0);
post_samples.Pi = cell(0);
post_samples.z = cell(0);
post_samples.par = cell(0);

% Pack parameters in par-struct
par.v0=v0;
par.p=p;
par.N=N;
par.alpha=alpha;
par.beta=beta;
par.gamma=gamma;
par.kappa=kappa;
par.begin_i=begin_i;
par.Sigma0=Sigma0_scale*Sigma0;
par.Sigma0_scale = Sigma0_scale;

switch par.emission_type
    case 'ZMG'
        emission_name = 'Zero Mean Gaussian (ZMG)';
    case 'SSM'
        emission_name = 'State Specific Mean (SSM)';
        par.lambda = lambda; % mean priors
        par.mu0 = mu0;
        post_samples.M = cell(0);
    case 'VAR'
        emission_name = 'Vector Auto Regressive (VAR)';
        post_samples.A = cell(0);
        R_inv = kron(diag(1./tau_M),eye(p));
        par.R_inv=R_inv;
        par.tau_M=tau_M;
        par.M = M;
        
        % Extract past based on model order of each process (M)
        Xpast = nan(p*M,N-n_tasks*M);
        wX = nan(p,N-n_tasks*M);
        nn = 0;
        begin_i_new = false(length(begin_i)-M*n_tasks,1);
        for ta = 1:n_tasks
            Xt = X(:,task_id_begin(ta):task_id_end(ta));
            Nt = size(Xt,2);
            for t = 1:M
                ids = (1+p*(t-1)):(1+p*(t-1))+p-1;
                Xpast(ids,1+nn:Nt-M+nn) = ...
                    Xt(:,(M-(t-1)):(Nt-t));
            end
            wX(:,(1:Nt-M)+nn) = Xt(:,(M+1):end);
            begin_i_new(nn+1) = true;
            nn = nn + Nt-M;
        end
        begin_i_old = begin_i;
        begin_i = begin_i_new;
        X = wX; clear wX
        Xtrue = X;
        N = N-n_tasks*M;
        par.N=N;
end

% Initialize relevant sufficient statistics and log probability of each
% cluster
[logP,par] = initializePar(X,z,par);
par.begin_i = begin_i;
par.const_state = const_state;
par.debug = debug;
par.debug_print = debug_print;
par.keyboard_interrupt = keyboard_interrupt;

% Initialize map-solution outputs
par_mj = par;
ll_map = -inf;

LL=nan(1,maxiter);
iters = 1:maxiter;

else
    opts = sampling_state.opts;
    par = sampling_state.par;
    logP = sampling_state.logP;
    z = sampling_state.z;
    maxiter = sampling_state.opts.maxiter;
    iters = sampling_state.iter:maxiter;
    LL = sampling_state.LL;
    post_samples = sampling_state.samples;
    par_mj = sampling_state.map.par;
    z_mj= sampling_state.map.z;
    ll_map = max(LL);
end

disp([' '])
disp(sprintf('Infinite Hidden Markov Model - %s',emission_name))
disp([' '])
disp(['To stop algorithm press control C'])
disp([' ']);
if nargin==3
   disp(['Restarting algorithm from previous state...Iteration: ' num2str(sampling_state.iter)]) 
end

%% Main loop
for iter=iters;
    iter_start = tic;
    % Gibbs sample
    if ~const_state && (iter<=burnin || ~max_annealing)
        [z,par,logP]=gibbs_sample(X,z,logP,par,false,randperm(N));
    elseif max_annealing && iter>burnin
        [z,par,logP]=gibbs_sample(X,z,logP,par,true,randperm(N));
    end
    
    % Gibbs sample change points
    if ~const_state && (iter<=burnin || ~max_annealing)
        c_points = find(z(1:end-1)~= z(2:end))';
        for rep = 1:cpoint_factor 
            [z,par,logP]=gibbs_sample(X,z,logP,par,false,c_points);
        end
    elseif max_annealing && iter>burnin
        c_points = find(z(1:end-1)~= z(2:end))';
        for rep = 1:cpoint_factor 
            [z,par,logP]=gibbs_sample(X,z,logP,par,true,c_points);
        end
    end
    
    % Split-merge sample
    if ~const_state && split_merge && (iter<=burnin || ~max_annealing)
        for rep=1:max(z)
            [z,par,logP]=split_merge_sample(X,z,logP,par,false);
        end
    elseif ~const_state && split_merge && max_annealing && iter>burnin
        for rep=1:max(z)
            [z,par,logP]=split_merge_sample(X,z,logP,par,true);
        end
    end
    
    % Sample hyper-parameters using Jurgen Van Gaels code from iHMM-v0.5__
    if ~const_state
        [par.beta, par.alpha, par.gamma] = iHmmHyperSample(par.N_trans, N, par.beta, par.alpha, par.gamma, hypers, 20);
    end
    
    % Random-walk kappa
    if ~const_kappa && ~const_state
        par = sample_kappa(par,z);
    end
    
    % Sample parameters from posterior (pre-marginalization)
    if mod(iter,post_thinning)==0 && iter>burnin && ~max_annealing
        disp('Sampling parameters from posterior...')
        post_samples = sampleParameterPosteriors(post_samples,z,par);
    end
    
    if const_state
        logM = 0;
    else
        logM=TransitionP(par,z);
    end
    
    % Reevaluate log-joint distribution
    K=max(z);
    LL(iter)=sum(logP)+logM;
    if ~max_annealing
        if LL(iter)>ll_map
            par_mj = par;
            ll_map = LL(iter);
            z_mj = z;
        end
    end
    if mod(iter,10)==0|| iter==1
        fprintf('%12s | %15s | %12s | %12s |','Iteration','Log-Likelihood','# of states','Time [s] \n');
    end
    fprintf('%12d | %15d | %12d | %12.4f | \n',iter,LL(iter),K,toc(iter_start));
    if iter==burnin && max_annealing
        disp(['  '])
        disp([' ... STARTING MAX ANNEALING ... '])
        disp(['  '])
    end
    
    if mod(iter,save_every)==0 && save_every~=0
        disp(' --- Saving state of inference procedure ---')
       sampling_state.par = par;
       sampling_state.z = z;
       sampling_state.LL = LL;
       sampling_state.logP = logP;
       sampling_state.map.par = par_mj;
       sampling_state.map.z = z_mj;
       sampling_state.opts = opts;
       sampling_state.iter = iter;
       sampling_state.samples = post_samples;
       save(save_file,'sampling_state')
    end
    
end

if max_annealing
    par_mj = par;
    z_mj = z;
end
% Expectation of parameters
E = calcExpectations(par);
%--------------------------------------------------------------------
function par=sample_kappa(par,z)
% Sample kappa - stickyness parameter (Random Walk MH)
if par.debug
    disp('Starting kappa sampler...')
end
maxiter = length(z);
logM = TransitionP(par,z);
accept=0;
for rep=1:maxiter
    kappa_new=exp(log(par.kappa)+0.1*randn);
    par_new = par; par_new.kappa = kappa_new;
    logM_new = TransitionP(par_new,z);
    if rand<(kappa_new/par.kappa*exp(logM_new-logM))
        %accept/reject
        par.kappa = kappa_new;
        logM = logM_new;
        accept=accept+1;
    end
end
if par.verbose
disp(['accepted ' num2str(accept) ' out of ' num2str(maxiter) ' samples for elements of kappa']);
end
%eof

%--------------------------------------------------------------------
function [z,par,logP]=split_merge_sample(X,z,logP,par,max_annealing)
% Split-merge sampling of state assignments
if par.debug && par.verbose
    disp('Starting split-merge sampler...')
end
p=par.p;
N=par.N;

switch par.emission_type
    case 'SSM'
        lambda=par.lambda;
        mu0=par.mu0;     
        
    case 'VAR'
        R_inv = par.R_inv;
end

i1=ceil(N*rand);
i2=ceil(N*rand);
while i2==i1
    i2=ceil(N*rand);
end

% swap such that z(i1) always has the cluster smaller index
% important because cluster 1 must always be kept valid!
if z(i1)>z(i2)
    i_temp = i1;
    i1 = i2;
    i2 = i_temp;
end

if z(i1)==z(i2) % Split move
    if par.debug && par.verbose
        disp('Considering Split move....')
    end
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
    
    [par_t,logdet] = calcEmissionSufficients(X,z_t,par,'split',comp);
    
    % Construct new state sequence statistics
    b_stick = betarnd(1, par.gamma);
    nlogL_beta = betalike([1, par.gamma],b_stick);
    par_t.beta(end+1) = (1-b_stick)*par_t.beta(end);
    par_t.beta(end-1) = b_stick*par_t.beta(end-1);
    par_t.N_trans = calcTransition(z_t,par);
    
    % calculate new log probability for each of the new clusters
    switch par.emission_type
        case 'ZMG'
            nn=(par_t.sumZ(comp(1))+par.v0);
            logP_t(comp(1))=par.logPrior-(nn/2*logdet(comp(1))-nn*p/2*log(2)-...
                mvgammaln(p,nn/2));
    
            nn=(par_t.sumZ(comp(2))+par.v0);
            logP_t(comp(2))=par.logPrior-(nn/2*logdet(comp(2))-nn*p/2*log(2)-...
                mvgammaln(p,nn/2));
        case 'SSM'
            nn=(par_t.sumZ(comp(1))+par.v0);
            logP_t(comp(1))=par.logPrior-(nn/2*logdet(comp(1))-nn*p/2*log(2)-...
                mvgammaln(p,nn/2))-p/2*(log(lambda)-log((par_t.sumZ(comp(1))+lambda)));
    
            nn=(par_t.sumZ(comp(2))+par.v0);
            logP_t(comp(2))=par.logPrior-(nn/2*logdet(comp(2))-nn*p/2*log(2)-...
                mvgammaln(p,nn/2))-p/2*(log(lambda)-log((par_t.sumZ(comp(2))+lambda)));
        case 'VAR'
            nn=(par_t.sumZ(comp(1))+par.v0);
            logP_t(comp(1)) = par.logPrior+logdet(comp(1))+nn*p/2*log(2)+mvgammaln(p,nn/2);
            
            nn=(par_t.sumZ(comp(2))+par.v0);
            logP_t(comp(2)) = par.logPrior+logdet(comp(2))+nn*p/2*log(2)+mvgammaln(p,nn/2);
    end
    
    % Restricted Gibbs sampling
    if sum(idx)>0
        for t=1:3
            [z_t,par_t,logP_t,logQ]=gibbs_sample(X,z_t,logP_t,par_t,false,find(idx)',comp);
        end
    else
        logQ=0;
    end
    logQ=logQ-nlogL_beta;
    logM_t=TransitionP(par_t, z_t);
    logM=TransitionP(par,z);
    
    %%% DEBUG %%%
        if par.debug
            joint_new=evalLogJoint(z_t,par_t);
            joint_old=evalLogJoint(z,par);
            tol_debug = 1e-8;
            if abs( sum(logP_t)+logM_t-sum(logP)-logM - (joint_new - joint_old))/max(abs(joint_new),abs(joint_old)) > tol_debug
                if par.debug_print
                   disp('Log-Joint Test Failed in Split-Merge Sampler') 
                end
                if par.keyboard_interrupt
                    keyboard
                end
            end
        end
    %%% DEBUG END %%%
    
    % Evalulate acceptance by MH-ratio
    if ~max_annealing && rand<exp(sum(logP_t)+logM_t-sum(logP)-logM-logQ);
        if par.verbose
            disp(['split component ' num2str(z(i1))]);
        end
        par=par_t;
        z=z_t;
        logP=logP_t;    
    elseif max_annealing && ((sum(logP_t)+logM_t-sum(logP)-logM-logQ)>0)
        if par.verbose
            disp(['split component ' num2str(z(i1))]);
        end
        par=par_t;
        z=z_t;
        logP=logP_t;
    end
else % merge move
    if par.debug && par.verbose
        disp('Considering Merge move....')
    end
    % Emission sufficients for merge move
    comp = [z(i1) z(i2)];
    [par_new,logdet] = calcEmissionSufficients(X,z,par,'merge',comp);
    
    % evaluate merge configuration
    idx = (z==z(i1) | z==z(i2));
    z_new=z;
    z_new(idx)=z(i1);
    z_new(z_new>z(i2))=z_new(z_new>z(i2))-1;
    
    % transition
    par_new.beta(z(i1))=par.beta(z(i1));
    par_new.beta(end)=par_new.beta(end)+par.beta(z(i2));
    par_new.beta(z(i2))=[];
    par_new.N_trans = calcTransition(z_new,par_new);

    logP_new=logP;
    switch par.emission_type
        case 'ZMG'
            nn=(par_new.sumZ(comp(1))+par.v0);
            logP_new(comp(1))=par.logPrior-(nn/2*logdet(comp(1))-nn*p/2*log(2)...
                -mvgammaln(p,nn/2));
        case 'SSM'
            nn=(par_new.sumZ(comp(1))+par.v0);
            logP_new(comp(1))=par.logPrior-(nn/2*logdet(comp(1))-nn*p/2*log(2)...
                -mvgammaln(p,nn/2))-p/2*(log(lambda)-log((par_new.sumZ(comp(1))+lambda)));
        case 'VAR'
            nn=(par_new.sumZ(comp(1))+par.v0);
            logP_new(comp(1))=par.logPrior+logdet(comp(1))+nn*p/2*log(2)+mvgammaln(p,nn/2);
    end
    logP_new(comp(2))=[];
    
    %%% DEBUG %%%
    if any(par_new.N_trans(:)<0)
        disp('N_trans is ....')
        keyboard
    end
    %%% DEBUG END %%%
    
    
    logM_new=TransitionP(par_new,z_new);
    logM=TransitionP(par,z);
    
    % Trick: If merge is 'worse' than old configuration - reject
    % immediately. Else start restricted Gibbs Sweep
    accept_rate=rand;
    if (~max_annealing && accept_rate<exp(sum(logP_new)+logM_new-sum(logP)-logM)) ...
            || (max_annealing && ((sum(logP_new)-sum(logP))>0))
        % generate split configuration
        z_t=z;
        z_t(idx)=comp(ceil(2*rand(sum(idx),1)));
        z_t(i1)=z(i1);
        z_t(i2)=z(i2);
        N_trans = calcTransition(z_t,par);
        
        idx(i1)=false;
        idx(i2)=false;
        logP_t=logP;
        
        [par_t,logdet] = calcEmissionSufficients(X,z_t,par,'split',[z(i1),z(i2)]);
        par_t.N_trans = N_trans;
        
        switch par.emission_type
            case 'ZMG'
                nn=(par_t.sumZ(z(i1))+par.v0);
                logP_t(z(i1))=par.logPrior-(nn/2*logdet(z(i1))-nn*p/2*log(2)...
                    -mvgammaln(p,nn/2));
        
                nn=(par_t.sumZ(z(i2))+par.v0);
                logP_t(z(i2))=par.logPrior-(nn/2*logdet(z(i2))-nn*p/2*log(2)...
                    -mvgammaln(p,nn/2));
            case 'SSM'
                nn=(par_t.sumZ(z(i1))+par.v0);
                logP_t(z(i1))=par.logPrior-(nn/2*logdet(z(i1))-nn*p/2*log(2)...
                    -mvgammaln(p,nn/2))-p/2*(log(lambda)-log((par_t.sumZ(z(i1))+lambda)));
        
                nn=(par_t.sumZ(z(i2))+par.v0);
                logP_t(z(i2))=par.logPrior-(nn/2*logdet(z(i2))-nn*p/2*log(2)...
                    -mvgammaln(p,nn/2))-p/2*(log(lambda)-log((par_t.sumZ(z(i2))+lambda)));
            case 'VAR'
                nn=(par_t.sumZ(z(i1))+par.v0);
                logP_t(z(i1)) = par.logPrior+logdet(z(i1))+nn*p/2*log(2)+mvgammaln(p,nn/2);
                
                nn=(par_t.sumZ(z(i2))+par.v0);
                logP_t(z(i2)) = par.logPrior+logdet(z(i2))+nn*p/2*log(2)+mvgammaln(p,nn/2);
        end
        % Restricted Gibbs sampling
        if sum(idx)>0
            for t=1:2
                [z_t,par_t,logP_t]=gibbs_sample(X,z_t,logP_t,par_t,false,find(idx)',comp);
            end
            [z_t,par_t,logP_t,logQ]=gibbs_sample(X,z_t,logP_t,par_t,false,find(idx)',comp,z);
        else
            logQ=0;
        end
        nlogL_beta = betalike([1, par.gamma],par_t.beta(z(i2))/par_new.beta(end));
        logQ=logQ-nlogL_beta;
        
        %%% DEBUG %%%
        if par.debug
            e = abs(norm(logP-logP_t))<1e-8;
            if ~e
                disp('LogP and LogP_t are not equal...')
                if par.keyboard_interrupt
                    logP
                    logP_t
                    keyboard
                end
            end

            [joint_new,trans_new]=evalLogJoint(z_new,par_new);
            [joint_old,trans_old]=evalLogJoint(z,par);
            tol_debug = 1e-8;
            if abs( sum(logP_new)+logM_new-sum(logP)-logM - (joint_new - joint_old))/max(abs(joint_new),abs(joint_old)) > tol_debug
                if par.debug_print
                   disp('Log-Joint Test Failed in Split-Merge Sampler') 
                end
                if par.keyboard_interrupt
                    keyboard
                end
            end
        end
        %%% DEBUG END %%%
        
        % Evaluate MH-ratio and accept/reject
        if ~max_annealing && accept_rate<exp(sum(logP_new)+logM_new-sum(logP_t)-logM+logQ);
            if par.verbose
                disp(['merged component ' num2str(z(i1)) ' with component ' num2str(z(i2))]);
            end
            par=par_new;
            z=z_new;
            logP=logP_new;
        elseif max_annealing && ((sum(logP_new)+logM_new-sum(logP_t)-logM+logQ)>0)
            if par.verbose
                disp(['merged component ' num2str(z(i1)) ' with component ' num2str(z(i2))]);
            end
            par=par_new;
            z=z_new;
            logP=logP_new;
        end
    end
end
% remove empty clusters
idx_empty=sort(find(par.sumZ==0),'descend');
for ii = idx_empty
    if par.verbose
        disp(['Deleting cluster ' num2str(ii) ' - Its empty!'])
    end
    switch par.emission_type
        case 'ZMG'
            par.R_avg(:,:,idx_empty)=[];
        case 'SSM'
            par.R_avg(:,:,idx_empty)=[];
            par.x_avg(:,:,idx_empty)=[];
        case 'VAR'
            par.cSbb(:,:,ii) = [];
            par.Sxb(:,:,ii) = [];
            par.Sxx(:,:,ii) = [];
    end
    logP(ii)=[];
    par.sumZ(ii)=[];
    par.beta(end)=par.beta(end)+par.beta(ii);
    par.beta(ii)=[];
    z(z>ii)=z(z>ii)-1;
end
par.N_trans = calcTransition(z,par);
%eof

%----------------------------------------------------------------------------
function [z,par,logP,logQ]=gibbs_sample(X,z,logP,par,max_annealing,sample_idx,comp,forced)
% Gibbs sampler for state assignments
% NB! Also used for restricted gibbs sampling (using the comp variable)
global Xpast
if par.debug && nargin<7 && par.verbose
    disp('Starting Gibbs sampler...')
end
if nargin<8
    forced=[];
end
if nargin<7
    comp=[];
end
logQ=0;
% Unpack needed parameters
K=max(z);
Ttot=size(X,2);
sumZ=par.sumZ;
N_trans=par.N_trans;
p=par.p;

logPrior=par.logPrior;
v0=par.v0;

beta=par.beta;
alpha=par.alpha;
kappa=par.kappa;

begin_i=par.begin_i;

switch par.emission_type
    case 'ZMG'
        R_avg=par.R_avg;
        R0=par.R0;
        
        Rtmp=zeros(p,p,K);
    case 'SSM'
        R_avg=par.R_avg;
        x_avg=par.x_avg;
        R0=par.R0;
        mu0 = par.mu0;
        lambda=par.lambda;
        
        Rtmp=zeros(p,p,K);
        x_avg_tmp = zeros(p,K);

    case 'VAR'
        M = par.M;
        R_inv=par.R_inv;
        Sigma0 = par.Sigma0;
        cSbb = par.cSbb;
        Sxb = par.Sxb;
        Sxx = par.Sxx;
        
        cSbb_tmp = zeros(M*p,M*p,K);
        Sxb_tmp = zeros(p,M*p,K);
        Sxx_tmp = zeros(p,p,K);
end

% gibbs sample clusters
for i=sample_idx
    %%% Remove i'th observation from all variables
    % transiton matrix
    if i~=Ttot && ~begin_i(i+1)
        z_after=z(i+1);
        N_trans(z(i),z_after)=N_trans(z(i),z_after)-1;
    else
        z_after=[];
    end
    if ~begin_i(i)
        z_before=z(i-1);
        N_trans(z_before,z(i))=N_trans(z_before,z(i))-1;
    else
        % t=1 (and the elements defined by begin_i) is assumed to come from state one
        % (in-transition)
        N_trans(1,z(i)) = N_trans(1,z(i)) - 1;
        z_before=[];
    end
    
    % Remove from sufficent statistics 
    
    switch par.emission_type
        case 'ZMG' % Zero-Mean Gaussian
            [R_avg(:,:,z(i)),flag]=cholupdate(R_avg(:,:,z(i)),X(:,i),'-');
            if flag==1
                warning('Cholesky down-date failed: Full downdate...')
                R_avg(:,:,z(i)) = chol(X(:,z==z(i))*X(:,z==z(i)) - X(:,i)*X(:,i) + par.Sigma0 );
            end
            sumZ(z(i))=sumZ(z(i))-1;
            logP(z(i))=logPrior-((sumZ(z(i))+v0)*sum(log(diag(R_avg(:,:,z(i)))))...
                -(sumZ(z(i))+v0)*p/2*log(2)-mvgammaln(p,(sumZ(z(i))+v0)/2));
        
        case 'SSM' % State Specific Mean
            R_avg(:,:,z(i))=cholupdate(R_avg(:,:,z(i)),... % remove mean-part
                1/sqrt(sumZ(z(i))+lambda)*(x_avg(:,z(i)) + lambda*mu0),'+'); %... and yes the sign is correct
            sumZ(z(i))=sumZ(z(i))-1;
            x_avg(:,z(i)) = x_avg(:,z(i)) - X(:,i);

            [R_avg(:,:,z(i)),flag]=cholupdate(R_avg(:,:,z(i)),... % add new mean
                1/sqrt(sumZ(z(i))+lambda)*(x_avg(:,z(i)) + lambda*mu0),'-'); %... and yes the sign is correct
            if flag==1
                warning('Cholesky down-date failed: Full downdate...')
                 R_avg(:,:,z(i)) = chol(X(:,z==z(i))*X(:,z==z(i))' + par.Sigma0 +...
                    lambda*(mu0*mu0') - 1/(sumZ(z(i))+lambda)*...
                    (x_avg(:,z(i))+lambda*mu0)*(x_avg(:,z(i))+lambda*mu0)');
            end


            [R_avg(:,:,z(i)),flag]=cholupdate(R_avg(:,:,z(i)),X(:,i),'-'); % remove covariance of i'th observation
            if flag==1
                warning('Cholesky down-date failed: Full downdate...')
                S = X(:,z==z(i))*X(:,z==z(i))' + par.Sigma0 +...
                    lambda*(mu0*mu0') - 1/(sumZ(z(i))+lambda)*...
                    (x_avg(:,z(i))+lambda*mu0)*(x_avg(:,z(i))+lambda*mu0)';
                [R_avg(:,:,z(i))]=chol(S-(X(:,i)*X(:,i)'));
            end
            
            logP(z(i))=logPrior-((sumZ(z(i))+v0)*sum(log(diag(R_avg(:,:,z(i)))))...
                -(sumZ(z(i))+v0)*p/2*log(2)-mvgammaln(p,(sumZ(z(i))+v0)/2))...
                -p/2*(log(lambda)-log((sumZ(z(i))+lambda)));
            
        case 'VAR' % Vector Autoregressive
            sumZ(z(i))=sumZ(z(i))-1;
            cSbb(:,:,z(i)) = cholupdate(cSbb(:,:,z(i)),Xpast(:,i),'-');
            Sxb(:,:,z(i)) = Sxb(:,:,z(i)) - X(:,i)*Xpast(:,i)';
            Sxx(:,:,z(i)) = Sxx(:,:,z(i)) - X(:,i)*X(:,i)';
    
            % Update likelihood
            Sxbx =  Sxb(:,:,z(i))/cSbb(:,:,z(i));
            Shat = Sxx(:,:,z(i)) - Sxbx*Sxbx';
            vnew = v0+sumZ(z(i));
            logP(z(i))=logPrior-p*sum(log(diag(cSbb(:,:,z(i))))) - ...
                vnew*sum(log(diag(chol(Shat)))) + vnew*p/2*log(2) + mvgammaln(p,vnew/2);
    end
    %%% Evaluate the assignment of i'th observation to all K+1 states
    logdet=zeros(1,K+1);
    if isempty(comp)
        sample_comp=1:(K+1);
    else
        sample_comp=comp;
    end
    
    for k=sample_comp % Update sufficient statistics with current data point
        if k<=K
            switch par.emission_type
                case 'ZMG'
                    Rtmp(:,:,k)=cholupdate(R_avg(:,:,k),X(:,i),'+');
                    logdet(k)=2*sum(log(diag(Rtmp(:,:,k))));
                case 'SSM'
                    Rtmp(:,:,k)=cholupdate(R_avg(:,:,k),X(:,i),'+');
                    x_avg_tmp(:,k) = x_avg(:,k) + X(:,i);
                    Rtmp(:,:,k)=cholupdate(Rtmp(:,:,k),... % remove mean-part
                        1/sqrt(sumZ(k)+lambda)*(x_avg(:,k) + lambda*mu0),'+'); 
                    [Rtmp(:,:,k),flag]=cholupdate(Rtmp(:,:,k),... % add new mean
                        1/sqrt(sumZ(k)+1+lambda)*(x_avg_tmp(:,k)+ lambda*mu0),'-'); 
                    if flag==1
                     warning('Cholesky down-date failed: Full downdate...')
                     idx = k==z; idx(i)=false;
                     Rtmp(:,:,k) = chol( X(:,idx)*X(:,idx)' + par.Sigma0 +...
                            lambda*(mu0*mu0') - 1/(sumZ(k)+1+lambda)*...
                            (x_avg_tmp(:,k)+lambda*mu0)*(x_avg_tmp(:,k)+lambda*mu0)');
                    end
                    logdet(k)=2*sum(log(diag(Rtmp(:,:,k))));
                case 'VAR'
                    cSbb_tmp(:,:,k) = cholupdate(cSbb(:,:,k),Xpast(:,i),'+');
                    Sxb_tmp(:,:,k) = Sxb(:,:,k) + X(:,i)*Xpast(:,i)';
                    Sxx_tmp(:,:,k) = Sxx(:,:,k) + X(:,i)*X(:,i)';
                    Sxbx = Sxb_tmp(:,:,k)/cSbb_tmp(:,:,k);
                    Shat = Sxx_tmp(:,:,k) - Sxbx*Sxbx';
                    % Calculate new log-determinant contribution
                    logdet(k)=-p*log(det(cSbb_tmp(:,:,k))) - ...
                        (v0+sumZ(k)+1)*sum(log(diag(chol(Shat))));
            end
        else % Calculate new sufficient for new cluster
            switch par.emission_type
                case 'ZMG'
                    Rtmp(:,:,k)=cholupdate(R0,X(:,i),'+');
                    logdet(k)=2*sum(log(diag(Rtmp(:,:,k))));
                case 'SSM'
                    Rtmp(:,:,k)=cholupdate(R0,X(:,i),'+');
                    Rtmp(:,:,k)=cholupdate(Rtmp(:,:,k),1/sqrt(1+lambda)*(X(:,i)+lambda*mu0),'-');
                    x_avg_tmp(:,k) = X(:,i);
                    logdet(k)=2*sum(log(diag(Rtmp(:,:,k))));
                case 'VAR'
                    cSbb_tmp(:,:,k) = chol(R_inv + Xpast(:,i)*Xpast(:,i)');
                    Sxb_tmp(:,:,k) = X(:,i)*Xpast(:,i)';
                    Sxx_tmp(:,:,k) = X(:,i)*X(:,i)' + Sigma0;
                    Sxbx = Sxb_tmp(:,:,k)/cSbb_tmp(:,:,k);
                    Shat = Sxx_tmp(:,:,k) - Sxbx*Sxbx';
                    % Calculate new log-determinant contribution
                    logdet(k)=-p*sum(log(diag(cSbb_tmp(:,:,k)))) - ...
                        (v0+1)*sum(log(diag(chol(Shat))));
            end
        end
    end
    if isempty(comp) % Full Gibbs sweep
        logP(K+1)=0;
        nn=[(sumZ+1+v0) 1+v0];
        %% This code snippet is copy past with small modifications of iHMM-v0.5__ by Jurgen Van Gael
        % Compute the marginal probability for timestep t.
        r = ones(1, K+1);
        for k=1:K
            if ~isempty(z_before)
                if z_before==k
                    r(k) = r(k) * ( N_trans(z_before, k) + alpha*beta(k) + kappa );
                else
                    r(k) = r(k) * ( N_trans(z_before, k) + alpha*beta(k) );
                end
            else
                if k==1
                    r(k) = r(k) * ( N_trans(1, k) + alpha * beta(k) + kappa);
                else
                    r(k) = r(k) * ( N_trans(1, k) + alpha * beta(k) );
                end
            end
            
            if ~isempty(z_after)
                if ~isempty(z_before) && k ~= z_before && k~= z_after
                    r(k) = r(k) * ( N_trans(k, z_after) + alpha * beta(z_after) ) / ( sum(N_trans(k, :)) + alpha + kappa );
                elseif ~isempty(z_before) && k ~= z_before && k == z_after
                    r(k) = r(k) * ( N_trans(k, z_after) + alpha * beta(z_after) + kappa ) / ( sum(N_trans(k, :)) + alpha + kappa );
                elseif isempty(z_before) && k ~= 1 && k~= z_after
                    r(k) = r(k) * ( N_trans(k, z_after) + alpha * beta(z_after) ) / ( sum(N_trans(k, :)) + alpha + kappa );
                elseif isempty(z_before) && k ~= 1 && k == z_after
                    r(k) = r(k) * ( N_trans(k, z_after) + alpha * beta(z_after) + kappa) / ( sum(N_trans(k, :)) + alpha + kappa );
                elseif ~isempty(z_before) && k == z_before && k ~= z_after
                    r(k) = r(k) * ( N_trans(k, z_after) + alpha * beta(z_after) ) / ( sum(N_trans(k, :)) + 1 + alpha + kappa );
                elseif ~isempty(z_before) && k == z_before && k == z_after
                    r(k) = r(k) * ( N_trans(k, z_after) + 1 + alpha * beta(z_after) + kappa ) / ( sum(N_trans(k, :)) + 1 + alpha + kappa );
                elseif isempty(z_before) && k == 1 && k ~= z_after
                    r(k) = r(k) * ( N_trans(k, z_after) + alpha * beta(z_after) ) / ( sum(N_trans(k, :)) + 1 + alpha + kappa);
                elseif isempty(z_before) && k == 1 && k == z_after
                    r(k) = r(k) * ( N_trans(k, z_after) + 1 + alpha * beta(z_after) + kappa ) / ( sum(N_trans(k, :)) + 1 + alpha + kappa );
                end
            end
        end
        
        if ~isempty(z_after)
            r(K+1) = alpha^2 * beta(K+1)* beta(z_after)/(alpha+kappa);
        else
            r(K+1) = (alpha * beta(K+1)*(alpha + kappa) )/(alpha+kappa);
        end
        %% end code snippet
        
        % Evaluate change in log-probability for each cluster
        switch par.emission_type
            case 'ZMG'
                logPnew=logPrior-(nn/2.*logdet-nn*p/2*log(2)-mvgammaln(p,nn/2));
            case 'SSM'
                logPnew=logPrior-(nn/2.*logdet-nn*p/2*log(2)-mvgammaln(p,nn/2))...
                      -p/2*(log(lambda)-log(([sumZ+1 1]+lambda)));
            case 'VAR'
                logPnew=logPrior+logdet+nn*p/2*log(2)+mvgammaln(p,nn/2);
        end
        logDif=logPnew-logP+log(r);
        PP=exp(logDif-max(logDif));
        
        %%% DEBUG %%%
        if par.debug
            
            switch par.emission_type
                case 'VAR'
                    ss.cSbb = cSbb_tmp;
                    ss.Sxx = Sxx_tmp;
                    ss.Sxb = Sxb_tmp;
                    
                case {'ZMG','SSM'}
                    ss = struct();
                
            end
            passed = logJointTest_Gibbs(i,z,logDif,r,par,beta,comp,ss);
        end
        %%% DEBUG END %%%
        
    else % Restricted Gibbs sweep
        nn=sumZ(comp)+1+v0;
        %% This code snippet is copy past with small modifications of iHMM-v0.5__ by Jurgen Van Gael
        % Compute the marginal probability for timestep t.
        r = ones(1, K);
        for k=comp
            if ~isempty(z_before)
                if z_before==k
                    r(k) = r(k) * ( N_trans(z_before, k) + alpha*beta(k) + kappa );
                else
                    r(k) = r(k) * ( N_trans(z_before, k) + alpha*beta(k) );
                end
            else
                if k==1
                    r(k) = r(k) * ( N_trans(1, k) + alpha * beta(k) + kappa);
                else
                    r(k) = r(k) * ( N_trans(1, k) + alpha * beta(k) );
                end
            end
            
            if ~isempty(z_after)
                if ~isempty(z_before) && k ~= z_before && k~= z_after
                    r(k) = r(k) * ( N_trans(k, z_after) + alpha * beta(z_after) ) / ( sum(N_trans(k, :)) + alpha + kappa );
                elseif ~isempty(z_before) && k ~= z_before && k == z_after
                    r(k) = r(k) * ( N_trans(k, z_after) + alpha * beta(z_after) + kappa ) / ( sum(N_trans(k, :)) + alpha + kappa );
                elseif isempty(z_before) && k ~= 1 && k~= z_after
                    r(k) = r(k) * ( N_trans(k, z_after) + alpha * beta(z_after) ) / ( sum(N_trans(k, :)) + alpha + kappa );
                elseif isempty(z_before) && k ~= 1 && k == z_after
                    r(k) = r(k) * ( N_trans(k, z_after) + alpha * beta(z_after) + kappa) / ( sum(N_trans(k, :)) + alpha + kappa );
                elseif ~isempty(z_before) && k == z_before && k ~= z_after
                    r(k) = r(k) * ( N_trans(k, z_after) + alpha * beta(z_after) ) / ( sum(N_trans(k, :)) + 1 + alpha + kappa );
                elseif ~isempty(z_before) && k == z_before && k == z_after
                    r(k) = r(k) * ( N_trans(k, z_after) + 1 + alpha * beta(z_after) + kappa ) / ( sum(N_trans(k, :)) + 1 + alpha + kappa );
                elseif isempty(z_before) && k == 1 && k ~= z_after
                    r(k) = r(k) * ( N_trans(k, z_after) + alpha * beta(z_after) ) / ( sum(N_trans(k, :)) + 1 + alpha + kappa);
                elseif isempty(z_before) && k == 1 && k == z_after
                    r(k) = r(k) * ( N_trans(k, z_after) + 1 + alpha * beta(z_after) + kappa ) / ( sum(N_trans(k, :)) + 1 + alpha + kappa );
                end
            end
        end
        
        %% end code snippet
        
        % Evaluate change in log-probability for each cluster
        switch par.emission_type
            case 'ZMG'
                logPnew=logPrior-(nn/2.*logdet(comp)-nn*p/2*log(2)-mvgammaln(p,nn/2));
            case 'SSM'
                logPnew=logPrior-(nn/2.*logdet(comp)-nn*p/2*log(2)-mvgammaln(p,nn/2))...
                    -p/2*(log(lambda)-log( (sumZ(comp)+1)+lambda));
            case 'VAR'
                logPnew=logPrior+logdet(comp)+nn*p/2*log(2)+mvgammaln(p,nn/2);
        end
        
        logDif=logPnew-logP(comp)+log(r(comp));
        PP=exp(logDif-max(logDif));
        
        %%% DEBUG %%%
        if par.debug
            passed = logJointTest_Gibbs(i,z,logDif,r,par,beta,comp);
        end
        %%% DEBUG END %%%
    end
    
    
    % sample from posterior
    if isempty(comp)
        if ~max_annealing
            z(i)=find(rand<cumsum(PP/sum(PP)),1,'first');
        else
            z(i)=find(PP==max(PP),1,'first');
        end
        logP(z(i))=logPnew(z(i));
    else
        if ~isempty(forced)
            z(i)=forced(i);
            logQ=logQ+log(PP(z(i)==comp)/sum(PP));
        else
            if ~max_annealing
                z(i)=comp(find(rand<cumsum(PP/sum(PP)),1,'first'));
            else
                z(i)=comp(find(PP==max(PP),1,'first'));
            end
        end
        logP(z(i))=logPnew(comp==z(i));
    end
    
    % Update sufficient statistics with new assignments
    if z(i)>K
        % extend beta by stick breaking construction (see also code of iHMM-v0.5__ by Jurgen Van Gael)
        b_stick = betarnd(1, par.gamma);
        beta(end+1) = (1-b_stick)*beta(end);
        beta(end-1) = b_stick*beta(end-1);
        K=K+1;
        sumZ(z(i))=1;
    else
        sumZ(z(i))=sumZ(z(i))+1;
    end
    
    switch par.emission_type
        case 'ZMG'
            R_avg(:,:,z(i))=Rtmp(:,:,z(i));
        case 'SSM'
            R_avg(:,:,z(i))=Rtmp(:,:,z(i));
            x_avg(:,z(i)) = x_avg_tmp(:,z(i));
        case 'VAR'
            cSbb(:,:,z(i)) = cSbb_tmp(:,:,z(i));
            Sxb(:,:,z(i)) = Sxb_tmp(:,:,z(i));
            Sxx(:,:,z(i)) = Sxx_tmp(:,:,z(i));
    end

    % Delete empty clusters
    idx_empty=sort(find(sumZ==0),'descend');
    for ii = idx_empty
        if par.verbose
            disp(['Deleting cluster ' num2str(ii) ' - Its empty!'])
        end
        z(z>ii)=z(z>ii)-1;
        
        switch par.emission_type
        case 'ZMG'
            R_avg(:,:,ii)=[];
        case 'SSM'
            R_avg(:,:,ii) = [];
            x_avg(:,ii) = [];
        case 'VAR'
            cSbb(:,:,ii) = [];
            Sxb(:,:,ii) = [];
            Sxx(:,:,ii) = [];
        end

        logP(ii)=[];
        sumZ(ii)=[];
        beta(end)=beta(end)+beta(ii);
        beta(ii)=[];
        K=K-1;
    end
    % Update transition matrix lastly...
    N_trans = calcTransition(z,par);
    
end


% Pack par struct with new parameters
switch par.emission_type
case 'ZMG'
    par.R_avg=R_avg;
case 'SSM'
    par.R_avg=R_avg;
    par.x_avg=x_avg;    
case 'VAR'
    par.cSbb = cSbb;
    par.Sxb = Sxb;
    par.Sxx = Sxx;
end
    
par.sumZ=sumZ;
par.N_trans=N_trans;
par.beta=beta;
logP=logP(1:K);
%eof

%%---------------------------------------------
function [logP,par]=initializePar(X,z,par)
% Initalizes par struct ready for computation
global Xpast
p = par.p;

switch par.emission_type
    case 'ZMG'
        R0=chol(par.Sigma0);
        logPrior=par.v0*sum(log(diag(R0)))-par.v0*p/2*log(2)...
            -mvgammaln(p,par.v0/2);
        val=unique(z);
        R_avg=zeros(p,p,length(val));
        sumZ=zeros(1,length(val));
        logP=zeros(1,length(val));
        logdet=zeros(1,length(val));
        for k=1:length(val)
            idx=(z==val(k));
            sumZ(k)=sum(idx);
            R_avg(:,:,k)=chol(X(:,idx)*X(:,idx)'+par.Sigma0);
            logdet(k)=2*sum(log(diag(R_avg(:,:,k))));
            nn=(sumZ(k)+par.v0);
            logP(k)=logPrior-(nn/2*logdet(k)-nn*p/2*log(2)-mvgammaln(p,nn/2));
        end
        par.R_avg=R_avg;
        par.R0=R0;
        par.logPrior = logPrior;
        par.sumZ=sumZ;
        
    case 'SSM'        
        R0=chol(par.Sigma0 + par.lambda*par.mu0*par.mu0');
        logPrior=par.v0*sum(log(diag(chol(par.Sigma0))))-par.v0*p/2*log(2)...
            -mvgammaln(p,par.v0/2);
        val=unique(z);
        R_avg=zeros(p,p,length(val));
        x_avg=zeros(p,length(val));
        sumZ=zeros(1,length(val));
        logP=zeros(1,length(val));
        logdet=zeros(1,length(val));
        for k=1:length(val)
            idx=(z==val(k));
            sumZ(k)=sum(idx);
            x_avg(:,k) = sum(X(:,idx),2);
            R_avg(:,:,k)=chol(X(:,idx)*X(:,idx)'+par.Sigma0 + ...
                par.lambda*(par.mu0*par.mu0')-1/(sumZ(k)+par.lambda)*(x_avg(:,k)...
                + par.lambda*par.mu0)*(x_avg(:,k) + par.lambda*par.mu0)' );
            logdet(k)=2*sum(log(diag(R_avg(:,:,k))));
            nn=(sumZ(k)+par.v0);
            logP(k)=logPrior-(nn/2*logdet(k)-nn*p/2*log(2)-mvgammaln(p,nn/2))....
                -p/2*(log(par.lambda)-log((sumZ(k)+par.lambda)));
        end
        par.R_avg=R_avg;
        par.R0=R0;
        par.x_avg = x_avg;
        par.logPrior = logPrior;
        par.sumZ=sumZ;
        
    case 'VAR'
        M = par.M;
        R_inv = kron(diag(1./par.tau_M),eye(p));
        logPrior=p/2*sum(log(diag(R_inv)))+par.v0*sum(log(diag(chol(par.Sigma0))))...
            -par.v0*p/2*log(2)-mvgammaln(p,par.v0/2);
        val=unique(z);
        cSbb = zeros(p*M,p*M,length(val));
        Sxb = zeros(p,p*M,length(val));
        Sxx = zeros(p,p,length(val));
        sumZ=zeros(1,length(val));
        logP=zeros(1,length(val));
        logdet=zeros(1,length(val));
        
        for k=1:length(val)
            idx=(z==val(k));
            sumZ(k)=sum(idx);
            cSbb(:,:,k) = chol(Xpast(:,idx)*Xpast(:,idx)' + R_inv);
            Sxb(:,:,k) = X(:,idx)*Xpast(:,idx)';
            Sxx(:,:,k) = X(:,idx)*X(:,idx)'+par.Sigma0;
            Sxbx = Sxb(:,:,k)/cSbb(:,:,k);
            Shat = Sxx(:,:,k) - Sxbx*Sxbx';
            vnew = sumZ(k)+par.v0;
            logdet(k)=-p*sum(log(diag(cSbb(:,:,k))))-vnew*sum(log(diag(chol(Shat))));
            logP(k)=logPrior + logdet(k) + vnew*p/2*log(2)+mvgammaln(p,vnew/2);
        end
        
        par.cSbb = cSbb;
        par.Sxb = Sxb;
        par.Sxx = Sxx;
        par.R_inv = R_inv;
        par.logPrior = logPrior;
        par.sumZ=sumZ;
        
    otherwise
        error('Unknown Emission Model :: Use ZMG, SSM or VAR')     
end
% Calculate transition matrix
par.N_trans = calcTransition(z,par);
    

%%---------------------------------------------
function N_trans = calcTransition(z,par)
N = par.N;
begin_i = par.begin_i;
Z=sparse(z,1:N,ones(1,N),max(z),N);
N_trans=full(Z(:,1:end-1)*Z(:,2:end)');
begin_i_tmp=find(begin_i);
% all begin_i timepoints are assumed to come from state 1
for tt = 1:length(begin_i_tmp)
    N_trans(1,z(begin_i_tmp(tt))) =  N_trans(1,z(begin_i_tmp(tt))) +1;
end
begin_i_tmp(begin_i_tmp==1)=[];
N_trans=N_trans-Z(:,begin_i_tmp-1)*Z(:,begin_i_tmp)';


%%---------------------------------------------
function [par_new,logdet] = calcEmissionSufficients(X,z,par,split_or_merge,comp)
global Xpast
if nargin < 3
   split_or_merge = 'none';
   comp = [];
end
par_new = par;
    switch split_or_merge
        case 'merge'
            % Get indices to be merged
            idx = z==comp(1) | z==comp(2);
            sumZ_new=par.sumZ;
            sumZ_new(comp(1)) = sum(idx);
            sumZ_new(comp(2)) = [];
            par_new.sumZ=sumZ_new;
            
            % Calculate new emission sufficients
            switch par.emission_type
                case 'ZMG'
                    % Unpack
                    Sigma0 = par.Sigma0;
                    
                    % New sufficient
                    par_new.R_avg(:,:,comp(1))=chol( X(:,idx)*X(:,idx)'+Sigma0);
                    par_new.R_avg(:,:,comp(2))=[];

                    % Contribution to logJoint
                    logdet(comp(1))=2*sum(log(diag(par_new.R_avg(:,:,comp(1)))));
                    
                case 'SSM'
                    % Unpack
                    lambda = par.lambda;
                    mu0 = par.mu0;
                    Sigma0 = par.Sigma0;
                    
                    % New sufficients
                    par_new.x_avg(:,comp(1)) = sum(X(:,idx),2);
                    par_new.x_avg(:,comp(2)) = [];
                    par_new.R_avg(:,:,comp(1))=chol( X(:,idx)*X(:,idx)'+Sigma0...
                        + lambda*(mu0*mu0') - 1/(sumZ_new(comp(1))+lambda)*... 
                        (par_new.x_avg(:,comp(1))+lambda*mu0)*(par_new.x_avg(:,comp(1))+lambda*mu0)' );
                    par_new.R_avg(:,:,comp(2))=[];
                    
                    % Contribution to logJoint
                    logdet(comp(1))=2*sum(log(diag(par_new.R_avg(:,:,comp(1)))));
                
                case 'VAR'
                    % Unpack
                    R_inv = par.R_inv;
                    Sigma0 = par.Sigma0;
                    v0 = par.v0;
                    p = par.p;
                    
                    % New sufficients
                    par_new.cSbb(:,:,comp(1)) = chol(Xpast(:,z==comp(1))*Xpast(:,z==comp(1))'...
                        + Xpast(:,z==comp(2))*Xpast(:,z==comp(2))'+ R_inv);
                    par_new.cSbb(:,:,comp(2)) = [];
                    
                    par_new.Sxb(:,:,comp(1)) = X(:,z==comp(1))*Xpast(:,z==comp(1))'...
                        + X(:,z==comp(2))*Xpast(:,z==comp(2))';
                    par_new.Sxb(:,:,comp(2)) = [];
                    
                    par_new.Sxx(:,:,comp(1)) = X(:,z==comp(1))*X(:,z==comp(1))'...
                        + X(:,z==comp(2))*X(:,z==comp(2))' + Sigma0;
                    par_new.Sxx(:,:,comp(2)) = [];
                    
                    
                    % Contribution to logJoint
                    Sxbx = par_new.Sxb(:,:,comp(1) )/par_new.cSbb(:,:,comp(1) );
                    Shat = par_new.Sxx(:,:,comp(1)) - Sxbx*Sxbx';
        
                    logdet(comp(1))=-p*sum(log(diag(par_new.cSbb(:,:,comp(1))))) - ...
                    (v0+par_new.sumZ(comp(1)))*sum(log(diag(chol(Shat))));
            end
            
        case 'split'
            % Calc new state proportions
            idx1 = z==comp(1);
            idx2 = z==comp(2);
            sumZ_new=par.sumZ;
            sumZ_new(comp(1)) = sum(idx1);
            sumZ_new(comp(2)) = sum(idx2);
            par_new.sumZ=sumZ_new;
           
            switch par.emission_type 
                case 'ZMG'
                    % Unpack 
                    Sigma0 = par.Sigma0;
                    
                    % New sufficients
                    par_new.R_avg(:,:,comp(1))=chol(X(:,idx1)*X(:,idx1)'+Sigma0);
                    par_new.R_avg(:,:,comp(2))=chol(X(:,idx2)*X(:,idx2)'+Sigma0);
                    
                    % Log joint contribution
                    logdet(comp(1))=2*sum(log(diag(par_new.R_avg(:,:,comp(1)))));
                    logdet(comp(2))=2*sum(log(diag(par_new.R_avg(:,:,comp(2)))));
                case 'SSM'
                    % Unpack 
                    lambda = par.lambda;
                    mu0 = par.mu0;
                    Sigma0 = par.Sigma0;
                    
                    % New sufficients
                    par_new.x_avg(:,comp(1)) = sum(X(:,idx1),2);
                    par_new.x_avg(:,comp(2)) = sum(X(:,idx2),2);
                    par_new.R_avg(:,:,comp(1))=chol(X(:,idx1)*X(:,idx1)'+Sigma0...
                        + lambda*(mu0*mu0') - 1/(lambda+par_new.sumZ(comp(1)))*...
                        (par_new.x_avg(:,comp(1))+lambda*mu0)*(par_new.x_avg(:,comp(1))+lambda*mu0)' );
                    par_new.R_avg(:,:,comp(2))=chol(X(:,idx2)*X(:,idx2)'+Sigma0...
                        + lambda*(mu0*mu0') - 1/(lambda+par_new.sumZ(comp(2)))*...
                        (par_new.x_avg(:,comp(2))+lambda*mu0)*(par_new.x_avg(:,comp(2))+lambda*mu0)' );
                    
                    % Log joint contribution
                    logdet(comp(1))=2*sum(log(diag(par_new.R_avg(:,:,comp(1)))));
                    logdet(comp(2))=2*sum(log(diag(par_new.R_avg(:,:,comp(2)))));
                    
                case 'VAR'
                    % Unpack 
                    R_inv = par.R_inv;
                    Sigma0 = par.Sigma0;
                    v0 = par.v0;
                    p = par.p;
                    
                    
                    % New sufficients
                    par_new.cSbb(:,:,comp(1)) = chol(Xpast(:,idx1)*Xpast(:,idx1)' + R_inv);
                    par_new.Sxb(:,:,comp(1)) = X(:,idx1)*Xpast(:,idx1)';
                    par_new.Sxx(:,:,comp(1)) = X(:,idx1)*X(:,idx1)' + Sigma0;
    
                    par_new.cSbb(:,:,comp(2)) = chol(Xpast(:,idx2)*Xpast(:,idx2)' + R_inv);
                    par_new.Sxb(:,:,comp(2)) = X(:,idx2)*Xpast(:,idx2)';
                    par_new.Sxx(:,:,comp(2)) = X(:,idx2)*X(:,idx2)' + Sigma0;
                    
                    % Log joint contribution
                    Sxbx = par_new.Sxb(:,:,comp(1))/par_new.cSbb(:,:,comp(1));
                    Shat = par_new.Sxx(:,:,comp(1)) - Sxbx*Sxbx';
                    logdet(comp(1)) = -p*sum(log(diag(par_new.cSbb(:,:,comp(1))))) - ...
                    (v0+par_new.sumZ(comp(1)))*sum(log(diag(chol(Shat))));
                    
                    Sxbx = par_new.Sxb(:,:,comp(2))/par_new.cSbb(:,:,comp(2));
                    Shat = par_new.Sxx(:,:,comp(2)) - Sxbx*Sxbx';
                    logdet(comp(2)) = -p*sum(log(diag(par_new.cSbb(:,:,comp(2))))) - ...
                    (v0+par_new.sumZ(comp(2)))*sum(log(diag(chol(Shat))));
            end
            
            
            
        case 'none'
            
            
            
        otherwise
            error('Invalid option for split_merge variable')
            
    end

    

%%---------------------------------------------
function logM=TransitionP(par,z)
% code snippet modified from  iHMM-v0.5__ by Jurgen Van Gael
logM=0;
for k=1:max(z)
    if length(par.N_trans(k,:))+1 ~= length(par.beta)
        keyboard
    end
    kapvec = zeros(1,max(z)+1); kapvec(k) = par.kappa;
    R = [par.N_trans(k,:) 0] + par.alpha * par.beta + kapvec;
    ab = par.alpha * par.beta + kapvec;
    nzind = find(R ~= 0);
    % Add transition likelihood.
    logM = logM + gammaln(par.alpha + par.kappa) ...
        - gammaln(sum([par.N_trans(k,:) 0]) + par.alpha + par.kappa) ...
        + sum(gammaln( R(nzind)  )) ...
        - sum(gammaln( ab(nzind) ));
end
%eof

%%-----------------------------------------------
function post_samples = sampleParameterPosteriors(post_samples,z,par)
% Function samples parameter posterior and returns the sample struct
p = par.p;
K = max(z);

switch par.emission_type
    case 'ZMG'
      postSIGK = zeros(p,p,K);
        % Sample Sigma
        for k = 1:K
            S = par.R_avg(:,:,k)'*par.R_avg(:,:,k);
            if ~issymmetric(S)
                warning('Sufficient statistics for Sigma is not symmetric!')
                warning('Symmetricizing...')
                S = (S+S')/2;
            end
            postSIGK(:,:,k) = iwishrnd(S,par.v0+par.sumZ(k));
        end
        post_samples.S{end+1} = postSIGK;
        
    case 'SSM'
        mu0 = par.mu0;
        lambda = par.lambda;
        
        postMU = zeros(p,K);
        postSIGK = zeros(p,p,K);
        for k = 1:K
            % Sample Covariance
            S = par.R_avg(:,:,k)'*par.R_avg(:,:,k);
            postSIGK(:,:,k) = iwishrnd(S,par.v0+par.sumZ(k));
            
            % Sample Mean
            muk = 1/(par.sumZ(k)+lambda)*(par.x_avg(:,k) + lambda*mu0);
            postMU(:,k) = mvnrnd(muk',1/(lambda+par.sumZ(k))*postSIGK(:,:,k))';
        end
        post_samples.M{end+1} = postMU;
        post_samples.S{end+1} = postSIGK;
        
    case 'VAR'
        M = par.M;
        
        Shat = zeros(p,p,K);
        postSIGK = zeros(p,p,K);
        postA = zeros(p,p*M,K);
        % Sample Sigma
        for k = 1:K
            Sxbx = par.Sxb(:,:,k)/par.cSbb(:,:,k);
            Shat(:,:,k) = par.Sxx(:,:,k) - Sxbx*Sxbx';
            if ~issymmetric(Shat(:,:,k)) % due to numerical instabilities
                warning('Sufficient statistics for Sigma is not symmetric!')
                warning('Symmetricizing...')
                Shat(:,:,k) = (Shat(:,:,k)+Shat(:,:,k)')/2;
            end
            postSIGK(:,:,k) = iwishrnd(Shat(:,:,k),par.v0+par.sumZ(k));
        end
        
        % Sample A
        for k = 1:K
            postA(:,:,k) = sample_matrixNormal(par.Sxb(:,:,k)/par.cSbb(:,:,k)/par.cSbb(:,:,k)',...
                postSIGK(:,:,k),par.cSbb(:,:,k)^(-1)/par.cSbb(:,:,k)');
        end
        
        post_samples.S{end+1} = postSIGK;
        post_samples.A{end+1} = postA;
end

% Sample Transition matrix
if ~par.const_state
    trans = SampleTransitionMatrix(z,par.alpha*par.beta(1:end-1),par.kappa);
    post_samples.Pi{end+1} = trans;
end
% Save everything
post_samples.z{end+1} = z;
post_samples.par{end+1} = par;
%eof


%%-----------------------------------------------
function E = calcExpectations(par)
% Expectation of emission parameters
switch par.emission_type
    case 'ZMG'
        p = par.p;
        K = size(par.R_avg,3);
        E.S = zeros(p,p,K);
        for k = 1:K
            S = par.R_avg(:,:,k)'*par.R_avg(:,:,k);
            if ~issymmetric(S)
                warning('Sufficient statistics for Sigma is not symmetric!')
                warning('Symmetricizing...')
                S = (S+S')/2;
            end
            E.S(:,:,k) = S/(par.sumZ(k)-1);
        end
        
    case 'SSM'
        p = par.p;
        K = size(par.R_avg,3);
        
        E.S = zeros(p,p,K);
        E.M = zeros(p,K);
        for k = 1:K
            % Covariance
            S = par.R_avg(:,:,k)'*par.R_avg(:,:,k);
            if ~issymmetric(S)
                warning('Sufficient statistics for Sigma is not symmetric!')
                warning('Symmetricizing...')
                S = (S+S')/2;
            end
            E.S(:,:,k) = S/(par.sumZ(k)-1);
            
            % Mean
            E.M(:,k) = 1/(par.sumZ(k)+par.lambda)*(par.x_avg(:,k)...
                + par.lambda*par.mu0);
        end
        
    case 'VAR'
        p = par.p;
        K = size(par.Sxx,3);
        
        E.S = zeros(p,p,K);
        E.A = zeros(p,p*par.M,K);
        for k = 1:K
            % Covariance
            Sxbx = par.Sxb(:,:,k)/par.cSbb(:,:,k);
            Shat = par.Sxx(:,:,k) - Sxbx*Sxbx';
            if ~issymmetric(Shat) % due to numerical instabilities
                warning('Sufficient statistics for Sigma is not symmetric!')
                warning('Symmetricizing...')
                Shat = (Shat+Shat')/2;
            end
            E.S(:,:,k) = Shat/(par.sumZ(k)-1);
            
            % VAR coefficients
            E.A(:,:,k) = par.Sxb(:,:,k)/par.cSbb(:,:,k)/par.cSbb(:,:,k)';
        end
end
%eof

%%-----------------------------------------------
function [logProb,logM] = evalLogJoint(z,par)
global Xtrue Xpast

% Evaluates LogJoint excluding hyperparameters
v0=par.v0;
Sigma0=par.Sigma0;
p = par.p;
N = par.N;
begin_i = par.begin_i;

switch par.emission_type
    case 'ZMG'
        logPrior=v0/2*log(det(Sigma0)) - v0*p/2*log(2)-mvgammaln(p,v0/2);
        for k=1:max(z)
            idx=(z==k);
            sumZ(k)=sum(idx);
            R_avg(:,:,k)=chol(Xtrue(:,idx)*Xtrue(:,idx)'+Sigma0);
            logdet(k)=2*sum(log(diag(R_avg(:,:,k))));
            nn=(sumZ(k)+v0);
            logP(k)=logPrior-(nn/2*logdet(k)-nn*p/2*log(2)-mvgammaln(p,nn/2));
        end
    case 'SSM'
        logPrior=v0/2*log(det(Sigma0)) - v0*p/2*log(2)-mvgammaln(p,v0/2);
        lambda = par.lambda;
        mu0 = par.mu0;

        for k=1:max(z)
            idx=(z==k);
            sumZ(k)=sum(idx);
            x_avg(:,k) = sum(Xtrue(:,idx),2);
            R_avg(:,:,k)=chol(Xtrue(:,idx)*Xtrue(:,idx)'+Sigma0 + lambda*(mu0*mu0')...
                -1/(sumZ(k)+lambda)*(x_avg(:,k)+lambda*mu0)*(x_avg(:,k)+lambda*mu0)');
            logdet(k)=2*sum(log(diag(R_avg(:,:,k))));
            nn=(sumZ(k)+v0);
            logP(k)=logPrior-(nn/2*logdet(k)-nn*p/2*log(2)-mvgammaln(p,nn/2))...
                -p/2*(log(lambda)-log((sumZ(k)+lambda)));
        end
        
    case 'VAR'
        R_inv = par.R_inv;
        logPrior=p/2*sum(log(diag(R_inv)))+v0*sum(log(diag(chol(Sigma0))))...
            - v0*p/2*log(2)-mvgammaln(p,v0/2);
        
        for k=1:max(z)
            idx=(z==k);
            sumZ(k)=sum(idx);
            cSbb(:,:,k) = chol(Xpast(:,idx)*Xpast(:,idx)' + R_inv);
            Sxb(:,:,k) = Xtrue(:,idx)*Xpast(:,idx)';
            Sxx(:,:,k) = Xtrue(:,idx)*Xtrue(:,idx)' + Sigma0;
            Sxbx = Sxb(:,:,k)/cSbb(:,:,k);
            Shat = Sxx(:,:,k) - Sxbx*Sxbx';
            vnew = sumZ(k)+v0;
            logdet(k)=-p*sum(log(diag(cSbb(:,:,k))))-vnew*sum(log(diag(chol(Shat))));
            logP(k)=logPrior + logdet(k) + vnew*p/2*log(2)+mvgammaln(p,vnew/2);
        end

end


% Recalc N_trans
N_trans = calcTransition(z,par);

par_tmp = par;
par_tmp.N_trans = N_trans;

logM=TransitionP(par_tmp,z);

logProb=sum(logP)+logM;
if strcmp(par.emission_type,'VAR')
    logProb=logProb-sum(log(par.tau_M));
end
%eof

function passed = logJointTest_Gibbs(i,z,logDif,r,par,beta,comp,ss)
global Xtrue
% join-likelihood test
tol_test = 1e-8; % tolerance on log-likelihood test
par1 = par; par2 = par;
par1.beta = beta; par2.beta = beta;

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

if par.debug_print
    disp(['Number of groups are ' num2str(max(z))])
    disp(['Two chosen groups are: ' num2str([z_pos(i1), z_pos(i2)])])
end

% if new cluster is generated it steals all the
% end-stick-mass
if i1 > max(z)
    par1.beta(end+1) = 0;
end

if i2 > max(z)
    par2.beta(end+1) = 0;
end

if max(z1)<max(z) % if a cluster magically disappears, mass is added to end of stick
    par1.beta(z_pos(i1))=beta(z_pos(i1));
    par1.beta(end)=par1.beta(end)+beta(z(i));
    %par1.beta(z(i1))=beta(z(i1)) +beta(z(i));
    par1.beta(z(i))=[];
end

if max(z2)<max(z)
    par2.beta(z_pos(i2))=beta(z_pos(i2));
    par2.beta(end)=par2.beta(end)+beta(z(i));
    %par2.beta(z(i2)) = beta(z(i2)) + beta(z(i));
    par2.beta(z(i))=[];
end

if (length(par1.beta)~= max(z1)+1) ||(length(par2.beta)~= max(z2)+1)
    disp('Beta is wrong...')
    keyboard
end

[logJointTest_1,logTransTest_1] = evalLogJoint(z1,par1);
[logJointTest_2,logTransTest_2] = evalLogJoint(z2,par2);
if par.debug_print
    disp('Evaluating log-joint')
    logDif(i1)-logDif(i2)
    logJointTest_1-logJointTest_2
end
passed =abs(logDif(i1)-logDif(i2)-(logJointTest_1-logJointTest_2))/max(abs(logJointTest_1),abs(logJointTest_2)) < tol_test;
if ~passed
    disp(['Joint test failed in Gibbs sampler'])
    if par.keyboard_interrupt
        disp(['Transition difference full: ' num2str(logTransTest_1 - logTransTest_2)])
        disp(['Transition difference conditional: ' num2str(log(r(z_pos(i1))) - log(r(z_pos(i2))))])
        disp(['Observed model difference full: ' num2str(logJointTest_1 -logTransTest_1 - logJointTest_2 + logTransTest_2)])
        disp(['Observed model difference conditional: ' num2str(logDif(i1) - log(r(z_pos(i1))) - logDif(i2) + log(r(z_pos(i2))))])
        
        %%% EXTRA DEBUG - REMOVE!!!
        if i1<=max(z) && i2<=max(z) && nargin>7
            [~,par11] = initializePar(Xtrue,z1,par);
            [~,par22] = initializePar(Xtrue,z2,par);
            fprintf(' ------ CLUSTER %d ------', i1)
            ss.cSbb(:,:,i1)-par11.cSbb(:,:,i1)
            ss.Sxx(:,:,i1)-par11.Sxx(:,:,i1)
            ss.Sxb(:,:,i1)-par11.Sxb(:,:,i1)

            fprintf(' ------ CLUSTER %d ------', i2)
            ss.cSbb(:,:,i2)-par11.cSbb(:,:,i2)
            ss.Sxx(:,:,i2)-par11.Sxx(:,:,i2)
            ss.Sxb(:,:,i2)-par11.Sxb(:,:,i2)
        end
        keyboard
    end
end
%eof
