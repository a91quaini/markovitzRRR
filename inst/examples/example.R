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
  penalty_type = 'd',
  step_size_type = 'd',
  step_size_constant = .05e-1,
  max_iter = 500
)
end_time_markovitz <- Sys.time()
# plot objective function vs solver iterations
PlotMarkovitzRRRObjective(markovitz_solution)

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

## Results
# Print the execution times
cat("MarkovitzRRR execution time:", end_time_markovitz - start_time_markovitz, "\n")
cat("CVX execution time:", end_time_cvx - start_time_cvx, "\n")

# Print optimal values
cat("MarkovitzRRR optimal value = ", round(min(markovitz_solution$objective), 4), "\n")
cat("CVX optimal value = ", round(cvx_solution$value, 4), "\n")

cat("Distance between MarkovitzRRR and CVX solutions = ",
    round(sum((markovitz_solution$solution - cvx_solution$getValue(X))^2), 15), "\n")
