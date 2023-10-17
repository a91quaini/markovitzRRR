// Author: Alberto Quaini

#ifndef UTILS_H
#define UTILS_H

#include <RcppArmadillo.h>

// [[Rcpp::export]]
Rcpp::List MarkovitzRRRCpp(
  const arma::mat& R,
  arma::mat& X0,
  const double lambda1 = 0.,
  const double lambda2 = 0.,
  const char penalty_type = 'd',
  const char step_size_type = 'd',
  const double step_size_constant = -1.,
  const unsigned int max_iter = 10000,
  const double tolerance = -1.
);

// [[Rcpp::export]]
Rcpp::List MarkovitzRRRAltCpp(
  const arma::mat& R,
  const double tau,
  const double lambda = 0.,
  const unsigned int max_iter = 10000,
  const double tolerance = -1.
);

#endif
