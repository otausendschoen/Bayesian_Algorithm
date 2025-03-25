//
// Project: Hierarchical Bayesian Analysis of Radon Data: An Enhanced Hamiltonian Monte Carlo Approach
// Group: Oliver Tausendschön, Marvin Ernst, Victor Sobottka
// Class: Probabilistic Inference in Machine Learning
// 
// HIERARCHICAL MODEL - BASELINE: CENTERED PARAMETRIZATION
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
  real<lower=0> tau;
  vector[J] alpha;
  real beta;
  real<lower=0> sigma;
}
model {
  mu_alpha ~ normal(0, 10);
  tau ~ cauchy(0, 2.5);
  alpha ~ normal(mu_alpha, tau);
  beta ~ normal(0, 10);
  sigma ~ cauchy(0, 2.5);

  y ~ normal(alpha[county] + beta * x, sigma);
}
generated quantities {
  vector[N] y_rep;
  vector[N] log_lik;
  for (n in 1:N) {
    y_rep[n] = normal_rng(alpha[county[n]] + beta * x[n], sigma);
    log_lik[n] = normal_lpdf(y[n] | alpha[county[n]] + beta * x[n], sigma);
  }
}

