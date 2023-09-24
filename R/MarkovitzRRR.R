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
#' @param penalty_parameters `n_parameters`-dimensional vector of penalty
#' parameter values from smallest to largest.
#' @param max_iter numeric solver parameter indicating the maximum number of
#' iterations. Default is `2000`
#' @param step_size_type character indicating the type of step size: `'s'` for
#' square summable but not summable and `'c'` for constant step size, any other
#' character gives a summable vanishing step size. Default is `'s'`.
#' @param step_size_constant numeric constant determining the step size. Default
#' is `1.e-3`
#' @param tolerance numeric tolerance check for `||X_k+1 - X_k||_F^2 / N^2`.
#' Default is 1.e-15.
#' @param check_arguments boolean `TRUE` if you want to check function arguments;
#' `FALSE` otherwise. Default is `TRUE`.
#'
#' @return a list.
#'
#' @export
MarkovitzRRR = function(
    returns,
    penalty_parameters,
    max_iter = 10000,
    step_size_type = 's',
    step_size_constant = 1.e-3,
    tolerance = 1.e-15,
    check_arguments = TRUE
) {

  if (check_arguments) {

    CheckReturns(returns)
    stopifnot("`penalty_parameters` must be numeric" = is.numeric(penalty_parameters))
    stopifnot("`max_iter` must be numeric" = is.numeric(max_iter))
    stopifnot("`step_size_type` must be a character" = is.character(step_size_type))
    stopifnot("`step_size_constant` must be numeric" = is.numeric(step_size_constant))
    stopifnot("`tolerance` must be numeric" = is.numeric(tolerance))

  }

  return(.Call(`_markovitzRRR_MarkovitzRRRCpp`,
               returns,
               penalty_parameters,
               max_iter,
               step_size_type,
               step_size_constant,
               tolerance
  ))

}
