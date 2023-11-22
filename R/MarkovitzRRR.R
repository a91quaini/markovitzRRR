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
#' @param lambda1 a number indicating the penalty parameter associated
#' with the Nuclear penalty `lambda1 * ||R * X||_*` or `lambda1 * ||X||_*`;
#' see `penalty_type`.
#' If it is less than or equal to zero, optimization is carried on without this
#' penalty.
#' @param lambda2 a number indicating the penalty parameter associated
#' with the Ridge penalty `lambda2 * ||X||_F^2`.
#' If it is less than or equal to zero, optimization is carried on without this
#' penalty.
#' @param initial_solution `n_returns x n_returns`-dimensional matrix of initial
#' hedging weights. Defaults is a hollow matrix with `1/N` as off-diagonal elements.
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
#' @return a list containing: the optimal solution in `$solution`; the optimal
#' value in `$objective`; the optimal portfolio weights in `$weights`.
#' if at least one of `lambda1` and `lambda2` is positive, then the list
#' additionally contains: the number of iterations in `$iterations`;
#' the solver status check (indicating if the objective value decreased from the
#' value at the initial value) in `$is_improved`; the solver status
#' check (indicating if the objective value at the last solution equal to
#' the value at the best solution?) in `$is_converged`.
#'
#' @examples
#' # Example usage with real data
#' returns = markovitzRRR::returns[,-1]
#' result = MarkovitzRRR(returns, lambda1 = 0.1, lambda2 = 0.1)
#'
#' @export
MarkovitzRRR = function(
  returns,
  lambda1,
  lambda2,
  initial_solution = matrix(0, 0, 0),
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
    stopifnot("`lambda1` must be nonnegative" = lambda1 >= 0.)
    stopifnot("`lambda2` must be numeric" = is.numeric(lambda2))
    stopifnot("`lambda2` must be nonnegative" = lambda2 >= 0.)
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


#############################
######  MarkovitzRRR ########
#############################

#' Compute Markovitz RRR
#'
#' @name MarkovitzRRRAlt
#' @description Computes Markovitz RRR
#'
#' @param returns `n_observations x n_returns`-dimensional matrix of test asset
#' excess returns.
#' @param penalty_parameters in principle this is a `n_parameters`-dimensional
#' vector of penalty
#' parameter values from smallest to largest; for now it is just a positive number.
#' @param max_iter numeric solver parameter indicating the maximum number of
#' iterations. Default is `10000`.
#' @param tolerance numeric tolerance check for `||X_k+1 - X_k||_F^2 / N^2`.
#' If `tolerance <= 0`, no check is performed. Default is `-1.`, for no .
#' @param check_arguments boolean `TRUE` if you want to check function arguments;
#' `FALSE` otherwise. Default is `TRUE`.
#'
#' @return a list containing
#'
#' @export
MarkovitzRRRAlt = function(
  returns,
  penalty_parameters,
  max_iter = 10000,
  tolerance = -1.,
  check_arguments = TRUE
) {

  if (check_arguments) {

    CheckReturns(returns)
    stopifnot("`penalty_parameters` must be numeric" = is.numeric(penalty_parameters))
    stopifnot("`penalty_parameters` must be positive" = penalty_parameters > 0.)
    stopifnot("`max_iter` must be numeric" = is.numeric(max_iter))
    stopifnot("`tolerance` must be numeric" = is.numeric(tolerance))

  }

  return(.Call(`_markovitzRRR_MarkovitzRRRAltCpp`,
    returns,
    penalty_parameters,
    -1.,
    max_iter,
    tolerance
  ))

}
