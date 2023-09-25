
# markovitzRRR: Markovitz optimal portfolio via Reduced Rank Regression

<!-- badges: start -->
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![R-CMD-check](https://github.com/a91quaini/markovitzRRR/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/a91quaini/markovitzRRR/actions/workflows/R-CMD-check.yaml)
<!-- badges: end -->

Author: Alberto Quaini

Efficient implementation of Markovitz optimal portfolio selection via Reduced Rank Regression.

## Installation

### Building from source

To install the latest (possibly unstable) development version from
GitHub, you can pull this repository and install it from the `R` command
line via

```R
# if you already have package `devtools` installed, you can skip the next line
install.packages("devtools")
devtools::install_github("a91quaini/markovitzRRR")
```

Package `markovitzRRR` contains `C++` code that needs to be
compiled, so you may need to download and install the [necessary tools
for MacOS](https://cran.r-project.org/bin/macosx/tools/) or the
[necessary tools for
Windows](https://cran.r-project.org/bin/windows/Rtools/).


## Example

This is a basic example which shows you how to solve a common problem:

``` r
## simulate asset returns
set.seed(2)
n_assets = 20
n_obs = 100
mean_returns = rep(0, n_assets)
variance_returns = diag(1., n_assets)
returns = MASS::mvrnorm(n_obs, mean_returns, variance_returns)

# set penalty parameter lambda
lambda = .05

## compute Markovitz RRR solution
start_time_markovitz <- Sys.time()
# step_size_type can be:
# `'c'` for constant step size equal to `step_size_constant`;
# `'s'` for square summable but not summable given by `step_size_constant / (iteration + 1)`;
# `'p'` for modified Polyak given by `step_size_constant / ||subgradient||_F^2`;
# any other character gives a summable vanishing step size given by
# `step_size_constant / sqrt(iteration + 1)`.
markovitz_solution = MarkovitzRRR(
  returns,
  lambda,
  max_iter = 10000,
  step_size_type ='v',
  step_size_constant = .05e-1
)
end_time_markovitz <- Sys.time()
# solver status
markovitz_solution$status
# solution
markovitz_solution$solution

## compute CVX solution
X = CVXR::Variable(n_assets, n_assets)
cost = .5 * CVXR::sum_squares(returns - returns %*% X)
penalty = lambda * CVXR::norm_nuc(returns %*% X)
constraint = list(CVXR::diag(X) == 0)

problem = CVXR::Problem(CVXR::Minimize(cost + penalty), constraint)

# Measure execution time for CVX
start_time_cvx <- Sys.time()
cvx_solution = CVXR::solve(problem, reltol = 1e-8, abstol = 1e-8, num_iter = 10000)
end_time_cvx <- Sys.time()
# cvx solution
cvx_solution$getValue(X)

## Results
# Print the execution times
cat("MarkovitzRRR execution time:", end_time_markovitz - start_time_markovitz, "\n")
cat("CVX execution time:", end_time_cvx - start_time_cvx, "\n")

# Print optimal values
cat("MarkovitzRRR optimal value = ", round(markovitz_solution$objective_value[length(markovitz_solution$objective_value)], 4), "\n")
cat("CVX optimal value = ", round(cvx_solution$value, 4), "\n")
```

Execution time:
``` r
MarkovitzRRR execution time: 0.3132079 
CVX execution time: 15.2143 
```

Optimal value:
``` r
MarkovitzRRR optimal value =  821.6312
CVX optimal value =  821.6312
```



