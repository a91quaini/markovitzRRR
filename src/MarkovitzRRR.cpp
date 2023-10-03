// Author: Alberto Quaini

#include "MarkovitzRRR.h"
#include "solver.h"

Rcpp::List MarkovitzRRRCpp(
  const arma::mat& R,
  const double lambda,
  const char objective_type,
  const char penalty_type,
  const char step_size_type,
  const double step_size_constant,
  const unsigned int max_iter,
  const double tolerance
) {

  // set initial point `X0` to hollow matrix with 1/N on the off-diagonal
  const arma::mat X0 = arma::toeplitz(arma::join_cols(
    arma::vec::fixed<1>(arma::fill::zeros),
    arma::vec(R.n_cols-1, arma::fill::value(1./R.n_cols))
  ));

  // initialize data
  MarkovitzRRRSolver solver(
    R,
    X0,
    lambda,
    objective_type,
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
    Rcpp::Named("weights") = solver.ComputeOptimalPortfolioWeights()
    // Rcpp::Named("status") = solver.GetStatus(),
  );

}
