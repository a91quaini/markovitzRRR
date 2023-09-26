# Author: Alberto Quaini

## Check returns
# Checks that returns are conforming to the packgae implementation.
CheckReturns = function(returns) {

  stopifnot("`returns` must contain numeric values" = is.numeric(returns))
  stopifnot("`returns` contains more assets (columns) than observations (rows)" = nrow(returns) > ncol(returns))
  stopifnot("`returns` must not contain missing values (NA/NaN)" = !anyNA(returns))

}

#' Plot Markovitz RRR solver objective function evolution
#'
#' @name MarkovitzRRR
#' @description Plots the  MarkovitzRRR objective function value at each iteration
#'
#' @param results list containing a numeric vector `objective` with the
#' markovitzRRR objective function values at each iteration.
#'
#' @export
PlotMarkovitzRRRObjective = function(results) {

  if (!is.list(results) || !any(names(results) == "objective")) {
    stop("Input must be a list containing an 'objective' numeric vector.")
  }

  plot(
    x = seq_along(results$objective),
    y = results$objective,
    pch = 19,
    xlab = "iteration",
    ylab = "objective Function",
  )

}
