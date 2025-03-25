//
// Project: Hierarchical Bayesian Analysis of Radon Data: An Enhanced Hamiltonian Monte Carlo Approach
// Group: Oliver Tausendschön, Marvin Ernst, Victor Sobottka
// Class: Probabilistic Inference in Machine Learning
//
// HIERARCHICAL MODEL - LOG-SCALE REPARAMETERIZATION FOR SCALE PARAMETERS
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
  vector[J] alpha;
  real log_tau;   // log-transformed group-level scale
  real log_sigma; // log-transformed residual scale
}
transformed parameters {
  real<lower=0> tau;
  real<lower=0> sigma;
  tau = exp(log_tau);
  sigma = exp(log_sigma);
}
model {
  // Priors for location parameters
  mu_alpha ~ normal(0, 10);
  beta ~ normal(0, 10);
  
  // Priors on the log-scale for tau and sigma, which imply lognormal priors on the original scales
  log_tau ~ normal(0, 1);
  log_sigma ~ normal(0, 1);
  
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

