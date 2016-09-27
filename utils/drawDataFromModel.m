function X = drawDataFromModel(post_samples,Ndat)
if nargin < 2
    Ndat = 1;
end

innovation_lvl = 1e-6;
p = size(post_samples.S{1},1);
T = length(post_samples.z{1});
if isfield(post_samples,'A')
    M = ceil( size(post_samples.A{1},2)/p);
end

X = nan(p,T,Ndat);
for n = 1:Ndat
    z =  post_samples.z{end-(n-1)};
    S = post_samples.S{end-(n-1)};
    if isfield(post_samples,'A')
        A = post_samples.A{end-(n-1)};
        X_past = innovation_lvl*randn(p,M);
        model = 'VAR';
    elseif isfield(post_samples,'mu')
        MU = post_samples.mu{end-(n-1)};
        model = 'SSM';
    else
        model = 'ZMG';
    end
    
    par = post_samples.par{end-(n-1)};
    
    for t = 1:T
       switch model
           case 'ZMG'
               X(:,t,n) = mvnrnd(zeros(1,p),S(:,:,z(t)),1)';
           case 'SSM'
               X(:,t,n) = mvnrnd(MU(:,z(t))',S(:,:,z(t)),1)';
           case 'VAR'
               if par.begin_i(t)
                  X_past = innovation_lvl*randn(p,M); 
               end
               X(:,t,n) = mvnrnd( (A(:,:,z(t))*X_past)',S(:,:,z(t)),1)';
               X_past = X(:,t,n);
       end
        
    end
       
end



%eof
end