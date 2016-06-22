function X = sample_matrixNormal(M,U,V)
Uc = chol(U,'lower'); Vc = chol(V);
Y = randn(size(M));
X = M + Uc*Y*Vc;
