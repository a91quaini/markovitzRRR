# Author: Alberto Quaini

# required packages:
# install.packages("MASS")
# install.packages("CVXR")
# devtools::install_github("a91quaini/markovitzRRR")
# install.packages("microbenchmark)

# Set seed for reproducibility
set.seed(2)

# Simulate asset returns
n_assets <- 295
n_obs <- 250
mean_returns = rep(0, n_assets)
variance_returns = diag(1., n_assets)
returns = MASS::mvrnorm(n_obs, mean_returns, variance_returns)
returns = apply(returns, 2, function(x) x - mean(x))

# sv_returns = svd(returns)$d
# const = 2. / (min(sv_returns)^2 + max(sv_returns)^2)
# const = 2. / (max(sv_returns)^2)

# Set penalty parameter lambda
lambda1 = 0.05
lambda2 = 0.2
# tau = .5

## markovitzRRR
markovitzRRR_function = function() {
  return(MarkovitzRRR(
    returns,
    lambda1=0.1,
    lambda2=0.,
    penalty_type = 'd',
    step_size_type = 'c',
    step_size_constant = -1.,
    max_iter = 100,
    tolerance = -1e-12
  ))
  return(ParallelMarkovitzRRR(
    returns,
    initial_solution = matrix(0,0,0),
    lambda1=0.1,
    lambda2=0.0,
    penalty_type = 'd',
    step_size_type = 'c',
    step_size_constant = -1.,
    max_iter = 100,
    tolerance = -1e-12
  ))
}

# markovitzRRRalt_function = function() {
#   return(MarkovitzRRRAlt(
#     returns,
#     tau,
#     max_iter = 200,
#     tolerance = -1.
#   ))
# }

## CVXR
X = CVXR::Variable(n_assets, n_assets)
cost = 0.5 * CVXR::sum_squares(returns - returns %*% X)
penalty = lambda1 * CVXR::norm_nuc(returns %*% X) + lambda2 * CVXR::sum_squares(X)
constraint = list(CVXR::diag(X) == 0)
problem = CVXR::Problem(CVXR::Minimize(cost + penalty), constraint)
# problem_constr = CVXR::Problem(
#   CVXR::Minimize(cost),
#   list(CVXR::diag(X) == 0, CVXR::norm_nuc(returns %*% X) <= tau)
# )

# Define the CVXR function call as a function
cvxr_function = function() {
  return(CVXR::solve(problem, verbose = FALSE, reltol = 1e-8, abstol = 1e-12, num_iter = 10000))
}

# # # Define the CVXR function call as a function
# cvxr_constr_function = function() {
#   return(CVXR::solve(problem_constr, verbose=TRUE, reltol = 1e-8, abstol = 1e-12, num_iter = 10000))
# }


# check solutions
markovitzRRR_solution = markovitzRRR_function()
# markovitzRRRalt_solution = markovitzRRRalt_function()
cvxr_solution = cvxr_function()
# cvxr_constr_solution = cvxr_constr_function()
# plot(1:length(markovitzRRRalt_solution$objective), markovitzRRRalt_solution$objective)
# cvxr_constr_solution$value
PlotMarkovitzRRRObjective(markovitzRRR_solution)
# PlotMarkovitzRRRObjective(markovitzRRRalt_solution)


cat("MarkovitzRRR optimal value = ", round(min(markovitzRRR_solution$objective), 5), "\n")
# cat("MarkovitzRRRAlt optimal value = ", round(min(markovitzRRRalt_solution$objective), 5), "\n")
cat("CVX optimal value = ", round(cvxr_solution$value, 5), "\n")


cat("Distance between MarkovitzRRR and CVX solutions = ",
    round(sum((markovitzRRR_solution$solution - cvxr_solution$getValue(X))^2), 15), "\n")
# cat("Distance between MarkovitzRRRalt and CVX solutions = ",
#     round(sum((markovitzRRRalt_solution$solution - cvxr_solution$getValue(X))^2), 15), "\n")


# Run microbenchmark
mb = microbenchmark::microbenchmark(
  markovitzRRR = markovitzRRR_function(),
  CVXR = cvxr_function(),
  times = 10
)

# Print the benchmark results
print(mb)

# Access specific metrics if needed
# print(summary(mb))
print(boxplot(mb))



returns = markovitzRRR::returns[1:200,2:15]
n_penalty_parameters = 10
lower_penalty_parameter = 1. / nrow(returns)^2
upper_penalty_parameter = 1.5
lambda2_values = c(exp(seq(
  from=log(lower_penalty_parameter),
  to=log(upper_penalty_parameter),
  length.out=n_penalty_parameters
)))


start_time <- Sys.time()
results = markovitzRRR::ParallelMarkovitzRRR(
  returns, initial_solution = matrix(0, 0, 0), lambda1 = 0.1, lambda2_values = lambda2_values
)
end_time <- Sys.time()
writeLines(paste("Parallel Computation time =", end_time - start_time))

start_time <- Sys.time()
for (i in seq_along(lambda2_values)) {

  result = markovitzRRR::MarkovitzRRR(
    returns, initial_solution = matrix(0, 0, 0), lambda1 = 0.1, lambda2 = lambda2_values[i]
  )

}
end_time <- Sys.time()
writeLines(paste("Computation time =", end_time - start_time))


res = MarkovitzRRR(
  returns,
  lambda1=0.1,
  lambda2=0.,
  penalty_type = 'd',
  step_size_type = 'c',
  step_size_constant = -1.,
  max_iter = 100,
  tolerance = -1e-12
)
X = res$solution
E <- (returns - returns %*% X)
Psi <- 1 / pmax(apply(E, 2, var), .Machine$double.eps)
theta <- diag(Psi) %*% (diag(n_assets) - X)
one <- matrix(1, n_assets, 1)
denom <- as.numeric(t(one) %*% theta %*% one)
weightsGMV <- (theta %*% one) / denom
weightsGMV - res$weights
