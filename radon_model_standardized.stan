//
// Project: Hierarchical Bayesian Analysis of Radon Data: An Enhanced Hamiltonian Monte Carlo Approach
// Group: Oliver Tausendschön, Marvin Ernst, Victor Sobottka
// Class: Probabilistic Inference in Machine Learning
//
// HIERARCHICAL MODEL - STANDARDIZING PREDICTORS
//

data {
  int<lower=1> N;
  int<lower=1> J;
  int<lower=1, upper=J> county[N];
  vector[N] x;
  vector[N] y;
}
transformed data {
  real x_mean;
  real x_sd;
  vector[N] x_std;
  
  x_mean = mean(x);
  x_sd = sd(x);
  x_std = (x - x_mean) / x_sd;
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
  
  // Priors for scale parameters
  tau ~ cauchy(0, 2.5);
  sigma ~ cauchy(0, 2.5);
  
  // Prior for group-level effects
  alpha ~ normal(mu_alpha, tau);
  
  // Likelihood using the standardized predictor
  y ~ normal(alpha[county] + beta * x_std, sigma);
}
generated quantities {
  vector[N] log_lik;
  for (n in 1:N)
    log_lik[n] = normal_lpdf(y[n] | alpha[county[n]] + beta * x_std[n], sigma);
}

