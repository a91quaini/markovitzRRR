# simulate matrix returns
n_assets = 20
n_obs = 100
mean_returns = rep(0, n_assets)
variance_returns = diag(1., n_assets)
returns = MASS::mvrnorm(n_obs, mean_returns, variance_returns)

# set lambda
lambda = .05

# Markovitz RRR
results = MarkovitzRRR(
  returns,
  lambda,
  max_iter = 10000,
  step_size_type ='l',
  step_size_constant = .05e-1
)
results$status
results$objective_value[length(results$objective_value)]
plot(1:length(results$objective_value), results$objective_value)
results$solution


# CVX
X = CVXR::Variable(n_assets, n_assets)
cost = .5 * CVXR::sum_squares(returns - returns %*% X)
penalty = lambda * CVXR::norm_nuc(returns %*% X)
constraint = list(CVXR::diag(X) == 0)

problem = CVXR::Problem(CVXR::Minimize(cost + penalty), constraint)
cvx_solution = CVXR::solve(problem, reltol = 1e-8, abstol = 1e-8, num_iter = 10000)
cvx_solution$getValue(X)
cvx_solution$value
