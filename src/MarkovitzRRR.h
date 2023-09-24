// Author: Alberto Quaini

#ifndef UTILS_H
#define UTILS_H

#include <RcppArmadillo.h>

// [[Rcpp::export]]
Rcpp::List MarkovitzRRRCpp(
  const arma::mat& R,
  const double lambda,
  const unsigned int max_iter = 10000,
  const char step_size_type = 's',
  const double step_size_constant = 1.e-3,
  const double tolerance = 1.e-15
);

#endif
