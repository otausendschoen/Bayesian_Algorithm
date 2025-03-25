//
// Project: Hierarchical Bayesian Analysis of Radon Data: An Enhanced Hamiltonian Monte Carlo Approach
// Group: Oliver Tausendschön, Marvin Ernst, Victor Sobottka
// Class: Probabilistic Inference in Machine Learning
//
// HIERARCHICAL MODEL - ROBUST PRIORS
//

data {
  int<lower=1> N;
  int<lower=1> J;
  int<lower=1, upper=J> county[N];
  vector[N] x;
  vector[N] y;
}
parameters {
  real mu_alpha;
  real beta;
  real<lower=0> tau;
  vector[J] alpha;
  real<lower=0> sigma;
}
model {
  // Priors for location parameters
  mu_alpha ~ normal(0, 10);
  beta ~ normal(0, 10);
  
  // Robust priors for scale parameters: half-t with 3 degrees of freedom
  tau ~ student_t(3, 0, 2.5) T[0, ];
  sigma ~ student_t(3, 0, 2.5) T[0, ];
  
  // Prior for group-level effects
  alpha ~ normal(mu_alpha, tau);
  
  // Likelihood
  y ~ normal(alpha[county] + beta * x, sigma);
}
generated quantities {
  vector[N] log_lik;
  for (n in 1:N)
    log_lik[n] = normal_lpdf(y[n] | alpha[county[n]] + beta * x[n], sigma);
}

