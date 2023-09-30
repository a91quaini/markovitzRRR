// Author: Alberto Quaini

#ifndef UTILS_H
#define UTILS_H

#include <RcppArmadillo.h>

// [[Rcpp::export]]
Rcpp::List MarkovitzRRRCpp(
  const arma::mat& R,
  const double lambda,
  const char penalty_type = 'd',
  const char step_size_type = 'd',
  const double step_size_constant = 1.e-3,
  const unsigned int max_iter = 10000,
  const double tolerance = -1.
);

#endif
