function logP_test=wishartMM_predict(Xtest,n_test,Xtrain,n_train,par,samples)
% Calculates predictive likelihood for the Infinite Wishart Mixture Model
% NB! Requires all slices of Xtest to be positive-definite
logP_tmp=nan(size(Xtest,3),length(samples));
for ss=1:length(samples)
    s=samples(ss);    
    NN=[full(sum(sparse(s.z,1:length(s.z),ones(1,length(s.z))),2)); s.alpha];
    E_pi=NN'/sum(NN);    
    logP=nan(size(Xtest,3),max(s.z)+1);
    for k=1:max(s.z)+1        
        Phi_k=sum(Xtrain(:,:,s.z==k) ,3)+s.Sigma0;
        R=chol(Phi_k);
        v=sum(n_train(s.z==k))+par.v0;
        for i=1:size(Xtest,3)
            Rtest=chol(Phi_k+Xtest(:,:,i));
            logP(i,k)=(n_test(i)-par.p-1)/2*log(det(Xtest(:,:,i)))-(v+n_test(i))*sum(log(diag(Rtest)))+mvgammaln(par.p,(v+n_test(i))/2)-mvgammaln(par.p,n_test(i)/2);
        end
        logP(:,k)=logP(:,k)-mvgammaln(par.p,v/2)+v*sum(log(diag(R))); % leftovers from the training-posterior
    end
    logP=bsxfun(@plus,logP,log(E_pi));
    maxlogP=max(logP,[],2);
    logP_tmp(:,ss)=log(sum(exp(bsxfun(@minus,logP,maxlogP)),2))+maxlogP;
end
maxlogP_tmp=max(logP_tmp,[],2);
logP_test=log(sum(exp(bsxfun(@minus,logP_tmp,maxlogP_tmp)),2))+maxlogP_tmp-log(length(samples));
    
    