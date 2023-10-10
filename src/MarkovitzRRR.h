// Author: Alberto Quaini

#ifndef UTILS_H
#define UTILS_H

#include <RcppArmadillo.h>
#include "constants.h"

// [[Rcpp::export]]
Rcpp::List MarkovitzRRRCpp(
  const arma::mat& R,
  arma::mat& X0,
  const double lambda = default_step_size_constant,
  const char penalty_type = default_choice_type,
  const char step_size_type = default_choice_type,
  const double step_size_constant = default_step_size_constant,
  const unsigned int max_iter = default_max_iter,
  const double tolerance = minus_one
);

// [[Rcpp::export]]
Rcpp::List MarkovitzRRRAltCpp(
  const arma::mat& R,
  const double tau,
  const double lambda = minus_one,
  const unsigned int max_iter = default_max_iter,
  const double tolerance = minus_one
);

#endif
