//
// Project: Hierarchical Bayesian Analysis of Radon Data: An Enhanced Hamiltonian Monte Carlo Approach
// Group: Oliver Tausendschön, Marvin Ernst, Victor Sobottka
// Class: Probabilistic Inference in Machine Learning
// 
// HIERARCHICAL MODEL - ADVANCED: NON-CENTERED PARAMETRIZATION
//

data {
  int<lower=1> N; 
  int<lower=1> J;
  int<lower=1> county[N];
  vector[N] y; 
  vector[N] x;
}

parameters {
  real beta; 
  real mu_alpha;
  real<lower=0> tau;
  vector[J] alpha_raw; 
  real<lower=0> sigma; 
}

transformed parameters {
  vector[J] alpha;
  for (j in 1:J) {
    alpha[j] = mu_alpha + alpha_raw[j] * tau;
  }
}

model {
  beta ~ normal(0, 10);
  mu_alpha ~ normal(0, 10);
  tau ~ cauchy(0, 2.5);
  sigma ~ cauchy(0, 2.5);
  alpha_raw ~ normal(0, 1); 

  for (n in 1:N) {
    y[n] ~ normal(alpha[county[n]] + beta * x[n], sigma);
  }
}

generated quantities {
  vector[N] log_lik;
  for (n in 1:N)
    log_lik[n] = normal_lpdf(y[n] | alpha[county[n]] + beta * x[n], sigma);
}




