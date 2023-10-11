# Author: Alberto Quaini

#############################
######  MarkovitzRRR ########
#############################

#' Compute Markovitz RRR
#'
#' @name MarkovitzRRR
#' @description Computes Markovitz RRR
#'
#' @param returns `n_observations x n_returns`-dimensional matrix of centred(!)
#' test asset excess returns.
#' @param penalty_parameter a positive number indicating the penalty parameter.
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
#' If it is negative, then it is internally set to
#' `2./(min(sv(R))^2 + max(sv(R))^2)`, where `sv` denotes singular values.
#' Default is `-1`
#' @param max_iter numeric solver parameter indicating the maximum number of
#' iterations. Default is `10000`.
#' @param tolerance numeric tolerance check for `||X_k+1 - X_k||_F^2 / N^2`.
#' If `tolerance <= 0`, no check is performed. Default is `-1`.
#' @param check_arguments boolean `TRUE` if you want to check function arguments;
#' `FALSE` otherwise. Default is `TRUE`.
#'
#' @return a list containing: the optimal solution in `$solution`; the optimal
#' value in `$objective`; the optimal portfolio weights in `$weights`; the
#' number of iterations in `$iterations`.
#'
#' @export
MarkovitzRRR = function(
  returns,
  penalty_parameter,
  initial_solution = matrix(0, 0, 0),
  penalty_type = 'd',
  step_size_type = 'd',
  step_size_constant = -1.,
  max_iter = 10000,
  tolerance = -1.,
  check_arguments = TRUE
) {

  if (check_arguments) {

    CheckReturns(returns)
    stopifnot("`initial_solution` must contain numeric values" = is.numeric(initial_solution))
    stopifnot("`penalty_parameter` must be numeric" = is.numeric(penalty_parameter))
    stopifnot("`penalty_parameter` must be positive" = penalty_parameter > 0.)
    stopifnot("`penalty_type` must be a character" = is.character(penalty_type))
    stopifnot("`step_size_type` must be a character" = is.character(step_size_type))
    stopifnot("`step_size_constant` must be numeric" = is.numeric(step_size_constant))
    stopifnot("`max_iter` must be numeric" = is.numeric(max_iter))
    stopifnot("`tolerance` must be numeric" = is.numeric(tolerance))


  }

  return(.Call(`_markovitzRRR_MarkovitzRRRCpp`,
    returns,
    initial_solution,
    penalty_parameter,
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
