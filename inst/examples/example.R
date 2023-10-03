# Author: Alberto Quaini

# required packages:
# install.packages("MASS")
# install.packages("CVXR")
# devtools::install_github("a91quaini/markovitzRRR")
# install.packages("microbenchmark)

# Set seed for reproducibility
set.seed(2)

# Simulate asset returns
n_assets <- 20
n_obs <- 100
mean_returns <- rep(0, n_assets)
variance_returns <- diag(1., n_assets)
returns <- MASS::mvrnorm(n_obs, mean_returns, variance_returns)

# Set penalty parameter lambda
lambda <- 0.05

# Define the markovitzRRR function call as a function
markovitzRRR_function <- function() {
  return(MarkovitzRRR(
    returns,
    lambda,
    objective_type = 'd',
    penalty_type = 'd',
    step_size_type = 'd',
    step_size_constant = 0.5e-2,
    max_iter = 10000,
     tolerance = 1e-12
  ))
}

X <- CVXR::Variable(n_assets, n_assets)
cost <- 0.5 * CVXR::sum_squares(returns - returns %*% X)
penalty <- lambda * CVXR::norm_nuc(returns %*% X)
constraint <- list(CVXR::diag(X) == 0)
problem <- CVXR::Problem(CVXR::Minimize(cost + penalty), constraint)

# Define the CVXR function call as a function
cvxr_function <- function() {
  return(CVXR::solve(problem, reltol = 1e-8, abstol = 1e-12, num_iter = 10000))
}

# check solutions
markovitzRRR_solution = markovitzRRR_function()
cvxr_solution = cvxr_function()

cat("MarkovitzRRR optimal value = ", round(min(markovitzRRR_solution$objective), 4), "\n")
cat("CVX optimal value = ", round(cvxr_solution$value, 4), "\n")

cat("Distance between MarkovitzRRR and CVX solutions = ",
    round(sum((markovitzRRR_solution$solution - cvxr_solution$getValue(X))^2), 15), "\n")


# Run microbenchmark
mb <- microbenchmark::microbenchmark(
  markovitzRRR = markovitzRRR_function(),
  CVXR = cvxr_function(),
  times = 10
)

# Print the benchmark results
print(mb)

# Access specific metrics if needed
# print(summary(mb))
print(boxplot(mb))

