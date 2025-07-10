# Author: Alberto Quaini

#############################
######  MarkovitzRRR ########
#############################

#' Compute Markovitz Optimal Portfolios via Reduced Rank Regression (RRR)
#'
#' @name MarkovitzRRR
#' @description Computes Markovitz optimal portfolios via Reduced Rank Regression
#' approach by solving the optimization problem:
#' minimize_X {0.5 ||R - RX||_F^2 + lambda1 ||RX||_* + lambda2/2 ||X||_F^2 | diag(X) = 0}
#' or the alternative
#' minimize_X {0.5 ||R - RX||_F^2 + lambda1 ||X||_* + lambda2/2 ||X||_F^2 | diag(X) = 0},
#' where ||.||_F denotes the Frobenious norm and ||.||_* the Nuclear norm.
#' Then, optimal weights are given by:
#' `w = diag(Var[E])^(-1) * (I - X)`, where `E` is the residual in the regression
#' R = RX + E.
#'
#' @param returns `n_observations x n_returns`-dimensional matrix of centred(!)
#' test asset excess returns.
#' @param initial_solution `n_returns x n_returns`-dimensional matrix of initial
#' hedging weights. If it is an empty matrix, then it is set to a hollow matrix
#' (a matrix with zero diagonal) with `1/N` as off-diagonal elements.
#' Default is `matrix(0, 0, 0)`.
#' @param lambda1 a number indicating the penalty parameter associated
#' with the Nuclear penalty `lambda1 * ||R * X||_*` or `lambda1 * ||X||_*`;
#' see `penalty_type`. Default is `0`.
#' If it is less than or equal to zero, optimization is carried on without this
#' penalty.
#' @param lambda2 a number indicating the penalty parameter associated
#' with the Ridge penalty `lambda2 * ||X||_F^2`.
#' If it is less than or equal to zero, optimization is carried on without this
#' penalty. Default is `0`.
#' @param penalty_type character indicating the type of penalty function: `'d'`
#' for default, i.e., penalty given by `||RX||_*`; `'a'` for alternative, i.e.,
#' penalty given by `||X||_*`. Default is `'d'`.
#' @param step_size_type character indicating the type of step size:
#' `'d'` for default, i.e., not summable vanishing:
#' `step_size_constant / sqrt(iter + 1)`;
#' `'s'` for square summable but not summable: `step_size_constant / (iter + 1)`;
#' `'l'` for constant step length: `step_size_constant / ||subgradient||_F`.
#' `'p'` for modified Polyak:
#' `(step_size_constant + objective_iter - min{objective_k | k=0,...,iter}) / ||subgradient)||_F`.
#' `'c'` for constant step size: `step_size_constant`.
#' Default is `'d'`.
#' @param step_size_constant numeric constant determining the step size.
#' If it is zero or negative, then it is internally set to
#' `2./(min(sv(R))^2 + max(sv(R))^2 + lambda2)`,
#' where `sv` denotes singular values. Default is `0`.
#' @param max_iter numeric solver parameter indicating the maximum number of
#' iterations. Default is `10000`.
#' @param tolerance numeric tolerance check for the Frobenious norm
#' of successive solutions `||X_k+1 - X_k||_F / N`.
#' If `tolerance > 0`, then the solver is stopped when `||X_k+1 - X_k||_F / N <= tolerance`.
#' If `tolerance <= 0`, no check is performed. Default is `0`.
#' @param check_arguments boolean `TRUE` if you want to check function arguments;
#' `FALSE` otherwise. Default is `TRUE`.
#'
#' @return a list containing: the optimal solution in `$solution`;
#' the estimated precision matrix in `$precision`; the optimal
#' value in `$objective`; the optimal portfolio weights in `$weights`; the number
#' of iterations in `$iterations`;
#' the solver status check (indicating if the objective value decreased from the
#' value at the initial value) in `$is_improved`; the solver status
#' check (indicating if the objective value at the last solution equal to
#' the value at the best solution?) in `$is_converged`.
#'
#' @examples
#' # Example usage with real data
#' returns = markovitzRRR::returns[1:50,-1]
#' result = MarkovitzRRR(returns, lambda1 = 0.1, lambda2 = 0.1)
#'
#' @export
MarkovitzRRR = function(
  returns,
  initial_solution = matrix(0, 0, 0),
  lambda1 = 0.,
  lambda2 = 0.,
  penalty_type = 'd',
  step_size_type = 'd',
  step_size_constant = -1.,
  max_iter = 10000,
  tolerance = 0.,
  check_arguments = TRUE
) {

  # Check arguments
  if (check_arguments) {

    CheckReturns(returns)
    stopifnot("`initial_solution` must contain numeric values" = is.numeric(initial_solution))
    stopifnot("`lambda1` must be numeric" = is.numeric(lambda1))
    stopifnot("`lambda2` must be numeric" = is.numeric(lambda2))
    stopifnot("`penalty_type` must be a character" = is.character(penalty_type))
    stopifnot("`step_size_type` must be a character" = is.character(step_size_type))
    stopifnot("`step_size_constant` must be numeric" = is.numeric(step_size_constant))
    stopifnot("`max_iter` must be numeric" = is.numeric(max_iter))
    stopifnot("`tolerance` must be numeric" = is.numeric(tolerance))

  }

  # Compute the Markovitz RRR solution
  return(.Call(`_markovitzRRR_MarkovitzRRRCpp`,
    returns,
    initial_solution,
    lambda1,
    lambda2,
    penalty_type,
    step_size_type,
    step_size_constant,
    max_iter,
    tolerance
  ))

}

########################################
######  ParallelMarkovitzRRRCpp ########
########################################

#' Compute Markovitz Optimal Portfolios via Reduced Rank Regression (RRR) in Parallel
#'
#' @name ParallelMarkovitzRRRCpp
#' @description This function computes Markovitz optimal portfolios via Reduced Rank
#' Regression approach in a parallel manner. It is designed to handle multiple values
#' of the Ridge penalty parameter (lambda2), solving the optimization problem for
#' each lambda2 value concurrently. The optimization problem is defined as:
#' minimize_X {0.5 ||R - RX||_F^2 + lambda1 ||RX||_* + lambda2/2 ||X||_F^2 | diag(X) = 0}
#' or the alternative
#' minimize_X {0.5 ||R - RX||_F^2 + lambda1 ||X||_* + lambda2/2 ||X||_F^2 | diag(X) = 0},
#' where ||.||_F denotes the Frobenious norm and ||.||_* the Nuclear norm.
#' The function then calculates the optimal weights as:
#' `w = diag(Var[E])^(-1) * (I - X)`, where `E` is the residual in the regression R = RX + E.
#'
#' @param returns `n_observations x n_returns`-dimensional matrix of centered(!)
#' test asset excess returns.
#' @param initial_solution `n_returns x n_returns`-dimensional matrix of initial hedging weights.
#' Defaults to a hollow matrix with `1/N` as off-diagonal elements.
#' Default is `matrix(0, 0, 0)`.
#' @param lambda1 a number indicating the penalty parameter associated
#' with the Nuclear penalty `lambda1 * ||R * X||_*` or `lambda1 * ||X||_*`;
#' see `penalty_type`. If less than or equal to zero, optimization is carried on without this penalty.
#' default is `0`.
#' @param lambda2_values a vector of numbers indicating the multiple values of the
#' Ridge penalty parameter `lambda2 * ||X||_F^2`. Each value of lambda2 is used in
#' a separate optimization problem solved in parallel. default is `0`.
#' @param penalty_type character indicating the type of penalty function: `'d'` for
#' default, i.e., penalty given by `||RX||_*`; `'a'` for alternative, i.e., penalty
#' given by `||X||_*`. Default is `'d'`.
#' @param step_size_type character indicating the type of step size: `'d'` for default,
#' i.e., not summable vanishing: `step_size_constant / sqrt(iter + 1)`; `'s'` for square
#' summable but not summable: `step_size_constant / (iter + 1)`; `'l'` for constant
#' step length: `step_size_constant / ||subgradient||_F`; `'p'` for modified Polyak:
#' `(step_size_constant + objective_iter - min{objective_k | k=0,...,iter}) / ||subgradient)||_F`;
#' `'c'` for constant step size: `step_size_constant`. Default is `'d'`.
#' @param step_size_constant numeric constant determining the step size.
#' If zero or negative, it is internally set to `2./(min(sv(R))^2 + max(sv(R))^2 + lambda2)`,
#' where `sv` denotes singular values. Default is `0`.
#' @param max_iter numeric solver parameter indicating the maximum number of iterations.
#' Default is `10000`.
#' @param tolerance numeric tolerance check for the Frobenious norm of successive
#' solutions `||X_k+1 - X_k||_F / N`. If `tolerance > 0`, the solver stops when
#' `||X_k+1 - X_k||_F / N <= tolerance`. If `tolerance <= 0`, no check is performed.
#' Default is `0`.
#' @param n_cores number of cores for parallel computation. Default is
#' `parallel::detectCores() - 1`, i.e., all available system cores minus one.
#' @param check_arguments boolean `TRUE` if you want to check function arguments;
#' `FALSE` otherwise. Default is `TRUE`.
#'
#' @return a list containing: the optimal solution in `$solution`;
#' the estimated precision matrix in `$precision`; the optimal
#' value in `$objective`; the optimal portfolio weights in `$weights`; the number
#' of iterations in `$iterations`;
#' the solver status check (indicating if the objective value decreased from the
#' value at the initial value) in `$is_improved`; the solver status
#' check (indicating if the objective value at the last solution equal to
#' the value at the best solution?) in `$is_converged`.
#'
#' @examples
#' # Example usage with real data and multiple lambda2 values
#' returns = markovitzRRR::returns[1:200,-1]
#' lambda2_values = c(0.1, 0.2, 0.3)
#' # results = markovitzRRR::ParallelMarkovitzRRR(
#' #   returns,
#' #   initial_solution = matrix(0, 0, 0),
#' #   lambda1 = 0.1,
#' #   lambda2_values = lambda2_values
#' #)
#'
#'
#' @export
ParallelMarkovitzRRR = function(
  returns,
  initial_solution = matrix(0, 0, 0),
  lambda1 = 0.,
  lambda2_values = 0.,
  penalty_type = 'd',
  step_size_type = 'd',
  step_size_constant = 0.,
  max_iter = 10000,
  tolerance = 0.,
  n_cores = parallel::detectCores() - 1,
  check_arguments = TRUE
) {

  # Check arguments
  if (check_arguments) {

    CheckReturns(returns)
    stopifnot("`initial_solution` must contain numeric values" = is.numeric(initial_solution))
    stopifnot("`lambda1` must be numeric" = is.numeric(lambda1))
    stopifnot("`lambda2_values` must be numeric" = is.numeric(lambda2_values))
    stopifnot("`penalty_type` must be a character" = is.character(penalty_type))
    stopifnot("`step_size_type` must be a character" = is.character(step_size_type))
    stopifnot("`step_size_constant` must be numeric" = is.numeric(step_size_constant))
    stopifnot("`max_iter` must be numeric" = is.numeric(max_iter))
    stopifnot("`tolerance` must be numeric" = is.numeric(tolerance))
    stopifnot("`n_cores` must be numeric" = is.numeric(n_cores))

  }

  # utils for parallel functionalities
  `%dopar%` = foreach::`%dopar%`
  cluster = parallel::makeCluster(
    min(n_cores, length(lambda2_values))
  )
  doParallel::registerDoParallel(cluster)

  # 1 C++ call per lambda2; each returns a list with 'solution' & 'precision'
  results <- foreach(
    idx = seq_len(length(lambda2_values)),
    .packages = "markovitzRRR"
  ) %dopar% {
    L <- .Call(
      `_markovitzRRR_ParallelMarkovitzRRRCpp`,
      returns,
      initial_solution,
      lambda1,
      lambda2_values[idx],
      penalty_type,
      step_size_type,
      step_size_constant,
      max_iter,
      tolerance
    )
    # keep only what we need
    list(
      lambda2    = lambda2_values[idx],
      solution   = L$solution,
      precision  = L$precision
    )
  }

  parallel::stopCluster(cluster)

  N <- ncol(returns)

  # build N×N×K arrays
  sol_array  <- simplify2array(lapply(results, `[[`, "solution"))
  prec_array <- simplify2array(lapply(results, `[[`, "precision"))

  # name the 3rd dimension with the lambda2 values
  lambda2s <- sapply(results, `[[`, "lambda2")
  dimnames(sol_array)[[3]]  <- lambda2s
  dimnames(prec_array)[[3]] <- lambda2s

  return(list(
    lambda2   = lambda2s,
    solution  = sol_array,   # N x N x K
    precision = prec_array   # N x N x K
  ))

}



# #' Compute Markovitz Optimal Portfolios via Reduced Rank Regression (RRR) in Parallel
# #'
# #' @name ParallelMarkovitzRRRCpp
# #' @description This function computes Markovitz optimal portfolios via Reduced Rank
# #' Regression approach in a parallel manner. It is designed to handle multiple values
# #' of the Ridge penalty parameter (lambda2), solving the optimization problem for
# #' each lambda2 value concurrently. The optimization problem is defined as:
# #' minimize_X {0.5 ||R - RX||_F^2 + lambda1 ||RX||_* + lambda2/2 ||X||_F^2 | diag(X) = 0}
# #' or the alternative
# #' minimize_X {0.5 ||R - RX||_F^2 + lambda1 ||X||_* + lambda2/2 ||X||_F^2 | diag(X) = 0},
# #' where ||.||_F denotes the Frobenious norm and ||.||_* the Nuclear norm.
# #' The function then calculates the optimal weights as:
# #' `w = diag(Var[E])^(-1) * (I - X)`, where `E` is the residual in the regression R = RX + E.
# #'
# #' @param returns `n_observations x n_returns`-dimensional matrix of centered(!)
# #' test asset excess returns.
# #' @param X0 `n_returns x n_returns`-dimensional matrix of initial hedging weights.
# #' Defaults to a hollow matrix with `1/N` as off-diagonal elements.
# #' @param lambda1 a number indicating the penalty parameter associated
# #' with the Nuclear penalty `lambda1 * ||R * X||_*` or `lambda1 * ||X||_*`;
# #' see `penalty_type`. If less than or equal to zero, optimization is carried on without this penalty.
# #' @param lambda2_values a vector of numbers indicating the multiple values of the
# #' Ridge penalty parameter `lambda2 * ||X||_F^2`. Each value of lambda2 is used in
# #' a separate optimization problem solved in parallel.
# #' @param penalty_type character indicating the type of penalty function: `'d'` for
# #' default, i.e., penalty given by `||RX||_*`; `'a'` for alternative, i.e., penalty
# #' given by `||X||_*`. Default is `'d'`.
# #' @param step_size_type character indicating the type of step size: `'d'` for default,
# #' i.e., not summable vanishing: `step_size_constant / sqrt(iter + 1)`; `'s'` for square
# #' summable but not summable: `step_size_constant / (iter + 1)`; `'l'` for constant
# #' step length: `step_size_constant / ||subgradient||_F`; `'p'` for modified Polyak:
# #' `(step_size_constant + objective_iter - min{objective_k | k=0,...,iter}) / ||subgradient)||_F`;
# #' `'c'` for constant step size: `step_size_constant`. Default is `'d'`.
# #' @param step_size_constant numeric constant determining the step size.
# #' If zero or negative, it is internally set to `2./(min(sv(R))^2 + max(sv(R))^2 + lambda2)`,
# #' where `sv` denotes singular values. Default is `0`.
# #' @param max_iter numeric solver parameter indicating the maximum number of iterations.
# #' Default is `10000`.
# #' @param tolerance numeric tolerance check for the Frobenious norm of successive
# #' solutions `||X_k+1 - X_k||_F / N`. If `tolerance > 0`, the solver stops when
# #' `||X_k+1 - X_k||_F / N <= tolerance`. If `tolerance <= 0`, no check is performed.
# #' Default is `0`.
# #'
# #' @return a list containing: a matrix of optimal portfolio weights for each lambda2 value
# #' in `$weights`; a vector indicating if the objective value decreased from the initial value
# #' for each lambda2 in `$is_improved`; a vector indicating if the objective value at the last
# #' solution equals the value at the best solution for each lambda2 in `$is_converged`.
# #'
# #' @examples
# #' # Example usage with real data and multiple lambda2 values
# #' returns = markovitzRRR::returns[,-1]
# #' lambda2_values = c(0.1, 0.2, 0.3)
# #' result = ParallelMarkovitzRRRCpp(
# #' returns, X0 = matrix(), lambda1 = 0.1, lambda2_values, penalty_type = 'd'
# #' )
# #'
# #' @export
# ParallelMarkovitzRRR = function(
#   returns,
#   initial_solution,
#   lambda1,
#   lambda2_values,
#   penalty_type = 'd',
#   step_size_type = 'd',
#   step_size_constant = -1.,
#   max_iter = 10000,
#   tolerance = 0.,
#   n_cores = 1,
#   check_arguments = TRUE
# ) {
#
#   # Check arguments
#   if (check_arguments) {
#
#     CheckReturns(returns)
#     stopifnot("`initial_solution` must contain numeric values" = is.numeric(initial_solution))
#     stopifnot("`lambda1` must be numeric" = is.numeric(lambda1))
#     stopifnot("`lambda2_values` must be numeric" = is.numeric(lambda2_values))
#     stopifnot("`penalty_type` must be a character" = is.character(penalty_type))
#     stopifnot("`step_size_type` must be a character" = is.character(step_size_type))
#     stopifnot("`step_size_constant` must be numeric" = is.numeric(step_size_constant))
#     stopifnot("`max_iter` must be numeric" = is.numeric(max_iter))
#     stopifnot("`tolerance` must be numeric" = is.numeric(tolerance))
#     stopifnot("`n_cores` must be numeric" = is.numeric(n_cores))
#
#   }
#
#   # Set the number of threads for parallel computation
#   RcppParallel::setThreadOptions(numThreads = num_cores)
#
#   # Compute the Markovitz RRR solutions
#   return(.Call(`_markovitzRRR_ParallelMarkovitzRRRCpp`,
#     returns,
#     initial_solution,
#     lambda1,
#     lambda2_values,
#     penalty_type,
#     step_size_type,
#     step_size_constant,
#     max_iter,
#     tolerance
#   ))
#
# }


# #############################
# ######  MarkovitzRRR ########
# #############################
#
# #' Compute Markovitz RRR
# #'
# #' @name MarkovitzRRRAlt
# #' @description Computes Markovitz RRR
# #'
# #' @param returns `n_observations x n_returns`-dimensional matrix of test asset
# #' excess returns.
# #' @param penalty_parameters in principle this is a `n_parameters`-dimensional
# #' vector of penalty
# #' parameter values from smallest to largest; for now it is just a positive number.
# #' @param max_iter numeric solver parameter indicating the maximum number of
# #' iterations. Default is `10000`.
# #' @param tolerance numeric tolerance check for `||X_k+1 - X_k||_F^2 / N^2`.
# #' If `tolerance <= 0`, no check is performed. Default is `-1.`, for no .
# #' @param check_arguments boolean `TRUE` if you want to check function arguments;
# #' `FALSE` otherwise. Default is `TRUE`.
# #'
# #' @return a list containing
# #'
# #' @export
# MarkovitzRRRAlt = function(
#   returns,
#   penalty_parameters,
#   max_iter = 10000,
#   tolerance = -1.,
#   check_arguments = TRUE
# ) {
#
#   if (check_arguments) {
#
#     CheckReturns(returns)
#     stopifnot("`penalty_parameters` must be numeric" = is.numeric(penalty_parameters))
#     stopifnot("`penalty_parameters` must be positive" = penalty_parameters > 0.)
#     stopifnot("`max_iter` must be numeric" = is.numeric(max_iter))
#     stopifnot("`tolerance` must be numeric" = is.numeric(tolerance))
#
#   }
#
#   return(.Call(`_markovitzRRR_MarkovitzRRRAltCpp`,
#     returns,
#     penalty_parameters,
#     -1.,
#     max_iter,
#     tolerance
#   ))
#
# }
