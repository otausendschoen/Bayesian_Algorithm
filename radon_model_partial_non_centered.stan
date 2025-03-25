//
// Project: Hierarchical Bayesian Analysis of Radon Data: An Enhanced Hamiltonian Monte Carlo Approach
// Group: Oliver Tausendschön, Marvin Ernst, Victor Sobottka
// Class: Probabilistic Inference in Machine Learning
//
// HIERARCHICAL MODEL - PARTIALLY NON-CENTERED PARAMETERIZATION
//

data {
  int<lower=1> N; 
  int<lower=1> J;
  int<lower=1, upper=J> county[N];
  vector[N] y; 
  vector[N] x;
}

parameters {
  real beta; 
  real mu_alpha;
  real<lower=0> tau;
  vector[J] alpha_raw; // non-centered component
  vector[J] delta;     // centered component
  real<lower=0> sigma; 
}

transformed parameters {
  real phi;  // fixed tuning parameter for partial non-centering
  phi = 0.5; // equal blending between centered and non-centered
  
  vector[J] alpha;
  for (j in 1:J) {
    // Combine non-centered and centered components
    alpha[j] = mu_alpha + phi * tau * alpha_raw[j] + (1 - phi) * delta[j];
  }
}

model {
  beta ~ normal(0, 10);
  mu_alpha ~ normal(0, 10);
  tau ~ cauchy(0, 2.5);
  sigma ~ cauchy(0, 2.5);
  
  // Priors for reparameterized parts
  alpha_raw ~ normal(0, 1);
  delta ~ normal(0, tau);
  
  for (n in 1:N) {
    y[n] ~ normal(alpha[county[n]] + beta * x[n], sigma);
  }
}

generated quantities {
  vector[N] log_lik;
  for (n in 1:N)
    log_lik[n] = normal_lpdf(y[n] | alpha[county[n]] + beta * x[n], sigma);
}

