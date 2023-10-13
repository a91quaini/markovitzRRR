// Author: Alberto Quaini

#include "MarkovitzRRR.h"
#include "solver.h"
#include "dykstra_ap.h"

Rcpp::List MarkovitzRRRCpp(
  const arma::mat& R,
  arma::mat& X0,
  const double lambda,
  const char penalty_type,
  const char step_size_type,
  const double step_size_constant,
  const unsigned int max_iter,
  const double tolerance
) {

  // if no initial `X0` is given, set it to hollow matrix with 1/N on the
  // off-diagonal
  if (!X0.n_elem) {

    X0 = arma::toeplitz(arma::join_cols(
      arma::vec::fixed<1>(arma::fill::zeros),
      arma::vec(R.n_cols-1, arma::fill::value(1./R.n_cols))
    ));

  }

  // initialize data
  MarkovitzRRRSolver solver(
    R,
    X0,
    lambda,
    penalty_type,
    step_size_type,
    step_size_constant,
    max_iter,
    tolerance
  );

  // solve the optimization problem
  solver.Solve();

  return Rcpp::List::create(
    Rcpp::Named("solution") = solver.GetSolution(),
    Rcpp::Named("objective") = solver.GetObjective(),
    Rcpp::Named("weights") = solver.GetWeights(),
    Rcpp::Named("iterations") = solver.GetIterations(),
    Rcpp::Named("X_norm") = solver.GetX_norm(),
    Rcpp::Named("Sigma_inv_norm") = solver.GetSigma_inv_norm(),
    Rcpp::Named("Weights") = solver.GetWWeights()
  );

}

Rcpp::List MarkovitzRRRAltCpp(
  const arma::mat& R,
  const double tau,
  const double lambda,
  const unsigned int max_iter,
  const double tolerance
) {

  // initialize data
  DykstraAP solver(
      R,
      tau,
      lambda,
      max_iter,
      tolerance
  );

  // solve the optimization problem
  solver.Solve();

  return Rcpp::List::create(
    Rcpp::Named("solution") = solver.GetSolution(),
    Rcpp::Named("objective") = solver.GetObjective()
    // Rcpp::Named("weights") = solver.ComputeOptimalPortfolioWeights()
    // Rcpp::Named("status") = solver.GetStatus(),
  );

}
