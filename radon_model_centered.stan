data {
  int<lower=1> N;              // number of observations
  int<lower=1> J;              // number of counties
  int<lower=1, upper=J> county[N]; // county index per observation
  vector[N] x;                 // floor (0 = basement, 1 = upper)
  vector[N] y;                 // log radon
}
parameters {
  real mu_alpha;              // overall intercept
  real<lower=0> tau;          // SD of county intercepts
  vector[J] alpha;            // county-specific intercepts
  real beta;                  // floor effect
  real<lower=0> sigma;        // residual SD
}
model {
  // Priors
  mu_alpha ~ normal(0, 10);
  tau ~ cauchy(0, 2.5);
  alpha ~ normal(mu_alpha, tau);
  beta ~ normal(0, 10);
  sigma ~ cauchy(0, 2.5);

  // Likelihood
  y ~ normal(alpha[county] + beta * x, sigma);
}
generated quantities {
  vector[N] y_rep;
  for (n in 1:N)
    y_rep[n] = normal_rng(alpha[county[n]] + beta * x[n], sigma);
}

