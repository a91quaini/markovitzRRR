# Author: Alberto Quaini

#############################
######  MarkovitzRRR ########
#############################

#' Compute Markovitz RRR
#'
#' @name MarkovitzRRR
#' @description Computes Markovitz RRR
#'
#' @param returns `n_observations x n_returns`-dimensional matrix of test asset
#' excess returns.
#' @param penalty_parameters in principle this is a `n_parameters`-dimensional
#' vector of penalty
#' parameter values from smallest to largest; for now it is just a positive number.
#' @param objective_type character indicating the type of objective function:
#' `'d'` for default, i.e., objective given by `.5||R - RX||_F^2 + lambda * ||RX||_*`;
#' `'a'` for alternative, i.e.,
#' objective given by `.5||R - RX||_F^2 + lambda * ||X||_*`. Default is `'d'`.
#' @param penalty_type character indicating the type of penalty function: `'d'`
#' for default, i.e., penalty given by `||RX||_*`; `'a'` for alternative, i.e.,
#' penalty given by `||X||_*`. Default is `'d'`.
#' @param step_size_type character indicating the type of step size:
#' `'c'` for constant step size equal to `step_size_constant`;
#' `'l'` for constant step length, where step size is given by
#' `step_size_constant / ||subgradient||_F`.
#' `'d'` for default, i.e., not summable vanishing step size given by
#' `step_size_constant / sqrt(iter + 1)`;
#' `'s'` for square
#' summable but not summable given by `step_size_constant / (iter + 1)`;
#' `'p'` for modified Polyak given by
#' `(step_size_constant + objective - min{objective_k | k=0,...,iter} ) / ||subgradient)||_F`.
#' Default is `'d'`.
#' @param step_size_constant numeric constant determining the step size. Default
#' is `1.e-3`
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
MarkovitzRRR = function(
  returns,
  penalty_parameters,
  objective_type = 'd',
  penalty_type = 'd',
  step_size_type = 'd',
  step_size_constant = 1.e-3,
  max_iter = 10000,
  tolerance = -1.,
  check_arguments = TRUE
) {

  if (check_arguments) {

    CheckReturns(returns)
    stopifnot("`penalty_parameters` must be numeric" = is.numeric(penalty_parameters))
    stopifnot("`penalty_parameters` must be positive" = penalty_parameters > 0.)
    stopifnot("`objective_type` must be a character" = is.character(objective_type))
    stopifnot("`penalty_type` must be a character" = is.character(penalty_type))
    stopifnot("`step_size_type` must be a character" = is.character(step_size_type))
    stopifnot("`step_size_constant` must be numeric" = is.numeric(step_size_constant))
    stopifnot("`max_iter` must be numeric" = is.numeric(max_iter))
    stopifnot("`tolerance` must be numeric" = is.numeric(tolerance))

  }

  return(.Call(`_markovitzRRR_MarkovitzRRRCpp`,
    returns,
    penalty_parameters,
    objective_type,
    penalty_type,
    step_size_type,
    step_size_constant,
    max_iter,
    tolerance
  ))

}
