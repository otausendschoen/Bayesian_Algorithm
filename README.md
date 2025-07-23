# Bayesian Hierarchical Modeling of Radon Levels

This project explores several Bayesian hierarchical models applied to county-level radon data using STAN and R. The goal is to estimate the effects of basement location and between-county variability in radon levels through partial pooling and model reparameterization.

---

## Project Summary

We analyze household radon measurements using a multilevel model to:
- Capture variation across counties.
- Assess the effect of basement presence on radon levels.
- Compare model performance under different parameterizations and priors.

---

## Model Structure

The primary model follows this form:

- **Observation model**:
  $y_i \sim \mathcal{N}(\alpha_{j[i]} + \beta x_i, \sigma_y)$

- **County-level intercepts**:
  $\alpha_j \sim \mathcal{N}(\mu_\alpha, \sigma_\alpha)$

Where:
- $x_i$: binary indicator for basement
- $j[i]$: county of observation $i$
- $\alpha_j$: county-level random effect

A full plate diagram is included in `baseline_model.jpeg`.

---

## Repository Structure

| File | Description |
|------|-------------|
| `Project.rmd` | Main analysis script including EDA, model fitting, diagnostics, and posterior plots |
| `radon_exported.csv` | Cleaned input data for modeling |
| `baseline_model.jpeg` | Hand-drawn diagram of the model plate notation |
| `.stan` files | STAN model definitions |
| `.rds` files | Fitted model objects saved after sampling |

---

## STAN Models Included

| Model | File (.stan) | Description |
|-------|--------------|-------------|
| Continuous baseline | `model_continuous.stan` | Non-hierarchical model with a shared intercept |
| Centered hierarchical | `radon_model_centered.stan` | Classic centered parameterization |
| Non-centered | `radon_model_non-centered.stan` | Improves convergence for weakly identified groups |
| Partial non-centered | `radon_model_partial_non_centered.stan` | Blend between centered and non-centered |
| Log-scale model | — | Uses log-transformed radon response |
| Standardized | `radon_model_standardized.stan` | Standardizes predictors and response |
| Robust model | `radon_model_robust.stan` | Uses t-distribution to handle outliers |



