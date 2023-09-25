# Author: Alberto Quaini

## Check returns
# Checks that returns are conforming to the packgae implementation.
CheckReturns = function(returns) {

  stopifnot("`returns` must contain numeric values" = is.numeric(returns))
  stopifnot("`returns` contains more assets (columns) than observations (rows)" = nrow(returns) > ncol(returns))
  stopifnot("`returns` must not contain missing values (NA/NaN)" = !anyNA(returns))

}

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
