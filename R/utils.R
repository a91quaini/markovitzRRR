# Author: Alberto Quaini

## Check returns
# Checks that returns are conforming to the packgae implementation.
CheckReturns = function(returns) {

  stopifnot("`returns` must contain numeric values" = is.numeric(returns))
  stopifnot("`returns` contains more assets (columns) than observations (rows)" = nrow(returns) > ncol(returns))
  stopifnot("`returns` must not contain missing values (NA/NaN)" = !anyNA(returns))

}
