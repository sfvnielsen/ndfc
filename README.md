# Infinite Hidden Markov Model and Infinite Wishart Mixture Model
Gibbs Sampler with Split-Merge Moves for the Infinite Hidden Markov Model (IHMM) (on top of Juergen Van Gaels IHMM toolbox http://mloss.org/software/view/205/ - see also the [Copyright Notice](#copyright-notice) below) and the Infinite Wishart Mixture Model (IWMM) implemented in MATLAB. The code was used a part of a publication "Predictive Assesment of Models for Dynamic Functional Connectivity", which as of now has been submitted. 


## Infinite Hidden Markov Model
The Infinite Hidden Markov Model (IHMM) [2] is the Bayesian non-parametric extension of the hidden Markov model (HMM), in which we place a prior on the number of states and through inference (MCMC) learn the posterior distribution over state sequences. 


### Emission Distributions


### Predictive Likelihood


## Infinite Wishart Mixture Model



## Copyright Notice
The following copyright notice has been copied from the IHMM implementation made by Juergen Van Gael (http://mloss.org/software/view/205/)

*(C) Copyright 2008-2009, Jurgen Van Gael (jv279 -at- cam .dot. ac .dot. uk)*

http://mlg.eng.cam.ac.uk/jurgen/

*Permission is granted for anyone to copy, use, or modify these
programs and accompanying documents for purposes of research or
education, provided this copyright notice is retained, and note is
made of any changes that have been made.*

*These programs and documents are distributed without any warranty,
express or implied.  As the programs were written for research
purposes only, they have not been tested to the degree that would be
advisable in any important application.  All use of these programs is
entirely at the user's own risk.*

### Changes from original code 
In our main Gibbs sweep (in IHMMgibbs.m) we have used a code snippet from iHmmSampleGibbs.m that calculates Dirichlet Process contribution to the conditional density. It has been indicated with comments in the code where this snippet appears. 

The following MATLAB functions have been directly copied to the current project
  - dirichlet_sample.m
  - SampleTransitionMatrix.m

All other code is original work. 

## References
[1] Van Gael, J. (July, 2010). The Infinite Hidden Markov Model 0.5. Retrieved from http://mloss.org/software/view/205/

[2] Beal, M. J., Ghahramani, Z., & Rasmussen, C. E. (2002). The Infinite Hidden Markov Model. In T. G. Dietterich and S. Becker and Z. Ghahramani (Ed.), Advances in Neural Information Processing Systems 14 (pp. 577–584). MIT Press.

[3] Nielsen, S. F. V., Madsen, K. H., Røge, R., Schmidt, M. N., & Mørup, M. (2016, January 4). Nonparametric Modeling of Dynamic Functional Connectivity in fMRI Data. arXiv [stat.AP]. Retrieved from http://arxiv.org/abs/1601.00496

[4] Korzen, J., Madsen, K. H., & Mørup, M. (June 8-12, 2014). Quantifying Temporal States in rs-fMRI Data using Bayesian Nonparametrics. Presented at the Human Brain Mapping 2014

