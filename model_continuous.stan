//
// Project: Hierarchical Bayesian Analysis of Radon Data: An Enhanced Hamiltonian Monte Carlo Approach
// Group: Oliver Tausendschön, Marvin Ernst, Victor Sobottka
// Class: Probabilistic Inference in Machine Learning
//
// HMC/NUTS for Continuous Parameters (Conditional on Discrete States)
// File: model_continuous.stan
//

data {
  int<lower=1> N;
  int<lower=1> J;
  int<lower=1, upper=J> county[N]; 
  vector[N] x;
  vector[N] y;  
  int<lower=1> Z[N];
}
transformed data {
  vector[N] Z_real;
  Z_real = rep_vector(0.0, N); 
  for (n in 1:N) {
    Z_real[n] = Z[n];  
  }
}
parameters {
  real mu_alpha; 
  real beta;
  real<lower=0> tau;
  vector[J] alpha; 
  real<lower=0> sigma;  
}
model {

  mu_alpha ~ normal(0, 10);
  beta ~ normal(0, 10);
  tau ~ cauchy(0, 2.5);
  sigma ~ cauchy(0, 2.5);
  alpha ~ normal(mu_alpha, tau);
  
  // Likelihood: use Z_real for proper real arithmetic
  y ~ normal(alpha[county] + beta * x + 0.1 * Z_real, sigma);
}
generated quantities {
  vector[N] log_lik;
  for (n in 1:N)
    log_lik[n] = normal_lpdf(y[n] | alpha[county[n]] + beta * x[n] + 0.1 * Z_real[n], sigma);
}

