# Non-parametric Dynamic Functional Connectivity (NDFC) Software in MATLAB
Gibbs Sampler with Split-Merge Moves for the Infinite Hidden Markov Model (IHMM) (on top of Juergen Van Gaels IHMM toolbox http://mloss.org/software/view/205/ - see also the LICENSE file) and the Infinite Wishart Mixture Model (IWMM) implemented in MATLAB. The code was used a part of a publication "Predictive Assesment of Models for Dynamic Functional Connectivity", which as of now has been submitted. But models support calculating the predictive likelihood on a new data set, and this is showcased in the two demos demoIHMM.m and demo_wishartMM.m. 

*NB! The IWMM requires the test-set to have all postive-definite matrices to currently calculate the predictive likelihood. A future release will feature an option to drop the term involving the determinant of the test data, which will still allow for model comparision and paramter tuning.*  

When the time comes that this (hopefully) gets published please throw a reference to:

Nielsen, S. F. V., Schmidt, M. N., Madsen, K. H. & Mørup, M. (Nov. 2016). *Predictive Assesment of Models for Dynamic Functional Connectivity*,  Submitted

## Infinite Hidden Markov Model
The Infinite Hidden Markov Model (IHMM) [2] is the Bayesian non-parametric extension of the hidden Markov model (HMM), in which we place a prior on the number of states and through inference (MCMC) learn the posterior distribution over state sequences. This construction is also known as a hierarchical Dirichlet Process. The implementation can be found in IHMMgibbs.m and a demonstration of how to use the code in demoIHMM.m.

### Emission Distributions
To fully specify the IHMM we need an emission distribution. In software we have implemented three types of emission namely,
* __ZMG__ : A zero-mean Gaussian in which the covariance matrix is state-specific. 
* __SSM__ : A Gaussian emission distribution with state specifc mean and covariance.
* __VAR__ : A Gaussian emission distribution with a state specific vector-autoregressive mean and covariance

To switch between emissions use the *opts* structure with flags *opts.emission_type = 'ZMG'* for the zero-mean Gaussian. All flags and settings can be seen in the header of the code.

## Infinite Wishart Mixture Model
The infinite Wishart Mixture Model (IWMM)[4] is the Bayesian non-parametric extensin of the Wishart Mixture Model [5], in which a clustering of scatter matrices is modeled using a mixture of Wishart distribution, i.e. the model does not work on the 'raw' data excatly but for instance on windowed covariance matrices. The implementation can be found in wishart_MM.m and a demonstration of how to use the code can be found in demo_wishartMM.m  


## References
[1] Van Gael, J. (July, 2010). The Infinite Hidden Markov Model 0.5. Retrieved from http://mloss.org/software/view/205/

[2] Beal, M. J., Ghahramani, Z., & Rasmussen, C. E. (2002). The Infinite Hidden Markov Model. In T. G. Dietterich and S. Becker and Z. Ghahramani (Ed.), Advances in Neural Information Processing Systems 14 (pp. 577–584). MIT Press.

[3] Nielsen, S. F. V., Madsen, K. H., Røge, R., Schmidt, M. N., & Mørup, M. (2016, January 4). Nonparametric Modeling of Dynamic Functional Connectivity in fMRI Data. arXiv [stat.AP]. Retrieved from http://arxiv.org/abs/1601.00496

[4] Korzen, J., Madsen, K. H., & Mørup, M. (June 8-12, 2014). Quantifying Temporal States in rs-fMRI Data using Bayesian Nonparametrics. Presented at the Human Brain Mapping 2014

[5] Hidot, S., & Saint-Jean, C. (2010). An Expectation–Maximization algorithm for the Wishart mixture model: Application to movement clustering. Pattern Recognition Letters, 31(14), 2318–2324.
