// Author: Alberto Quaini

#include "MarkovitzRRR.h"
#include "solver.h"

Rcpp::List MarkovitzRRRCpp(
  const arma::mat& R,
  const double lambda,
  const unsigned int max_iter,
  const char step_size_type,
  const double step_size_constant,
  const double tolerance
) {

  // set initial point X to hollow matrix with 1/N on the off-diagonal
  const arma::mat X_initial = arma::toeplitz(arma::join_cols(
    arma::vec::fixed<1>(arma::fill::zeros),
    arma::vec(R.n_cols-1, arma::fill::value(1./R.n_cols))
  ));

  // initialize data
  MarkovitzRRRSolver solver(
    R,
    X_initial,
    lambda,
    step_size_type,
    step_size_constant,
    tolerance
  );

  // set iteration counter to 1
  unsigned int iter = 1;

  // algorithm loop
  while(iter < max_iter) {

    //// Compute projected subgradient step:
    // compute step and G0
    // where G0 is an element in the subgradient of the objective function at X0
    // compute X1 = P(X0 - step * G0)
    // project on space of hollow matrices
    // check if f(X1) <= f(Xbest)
    solver.ProjectedSubgradientStep(iter);

    if (solver.GetStatus() == "solved") break;

    // increase iteration
    ++iter;

  }

  return Rcpp::List::create(
    Rcpp::Named("solution") = solver.GetSolution(),
    Rcpp::Named("objective") = solver.GetObjective(),
    Rcpp::Named("status") = solver.GetStatus(),
    Rcpp::Named("weights") = solver.GetOptimalPortfolioWeights()
  );

}
