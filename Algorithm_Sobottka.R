install.packages("here")
library(here)
library(rstan)
library(ggplot2)
library(dplyr)


set.seed(123)

radon_data <- read.csv(here("data", "radon_exported.csv"))

# Display the first few rows of the dataset
head(radon_data)

#number of houses
number_of_houses <- nrow(radon_data)
cat("Number of houses:", nrow(radon_data), "\n")

#number of counties
unique_counties <- unique(radon_data$county)
num_unique_counties <- length(unique_counties)
cat("Number of unique counties:", num_unique_counties, "\n")

# Filter data for floor 0 and floor 1, and count unique counties
counties_floor_0 <- mn_radon %>%
  filter(floor == 0) %>%
  summarise(unique_counties = n_distinct(county)) %>%
  pull(unique_counties)

counties_floor_1 <- mn_radon %>%
  filter(floor == 1) %>%
  summarise(unique_counties = n_distinct(county)) %>%
  pull(unique_counties)

# Print the results
print(paste('Counties with measurements from floor 0:', counties_floor_0))
print(paste('Counties with measurements from floor 1:', counties_floor_1))


# Prepare data for Stan model
stan_data <- list(
  N = nrow(radon_data),
  y = radon_data$log_radon,
  county = radon_data$county
)

# Define the Stan model code
stan_code <- "
data {
  int<lower=1> N;
  vector[N] x;
  vector[N] y;
}
parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;
}
model {
  y ~ normal(alpha + beta * x, sigma);
  alpha ~ normal(0, 10);
  beta ~ normal(0, 10);
  sigma ~ normal(0, 10);
}
generated quantities {
  array[N] real y_rep = normal_rng(alpha + beta * x, sigma);
}
"

# Compile the Stan model
stan_model <- stan_model(model_code = stan_code)

# Fit the model to the data
fit <- sampling(stan_model, data = stan_data, iter = 1000, chains = 4)

# Extract the results
fit_results <- extract(fit)

# Plot the results using ggplot2
beta_samples <- fit_results$beta
ggplot(data = data.frame(beta = beta_samples), aes(x = beta)) +
  geom_density() +
  labs(title = "Posterior Distribution of Beta")
