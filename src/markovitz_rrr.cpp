// Author: Alberto Quaini

#include "markovitz_rrr.h"
#include "markovitz_rrr_solver.h"
#include "parallel_markovitz_rrr_solver.h"

////////////////////////////////
////// MarkovitzRRRCpp /////////

Rcpp::List MarkovitzRRRCpp(
  const arma::mat& returns,
  arma::mat& X0,
  const double lambda1,
  const double lambda2,
  const char penalty_type,
  const char step_size_type,
  const double step_size_constant,
  const unsigned int max_iter,
  const double tolerance
) {

  // if no initial `X0` is given, set it to hollow matrix with 1/N on the
  // off-diagonal
  if (X0.empty()) {

    X0 = arma::toeplitz(arma::join_cols(
      arma::vec::fixed<1>(arma::fill::zeros),
      arma::vec(returns.n_cols - 1, arma::fill::value(1. / returns.n_cols))
    ));

  }

  // initialize data
  MarkovitzRRRSolver solver(
    returns,
    X0,
    lambda1,
    lambda2,
    penalty_type,
    step_size_type,
    step_size_constant,
    max_iter,
    tolerance
  );

  // solve the optimization problem
  solver.Solve();

  // return output list
  return solver.GetOutputList();

}

////////////////////////////////////////
////// ParallelMarkovitzRRRCpp /////////

arma::rowvec ParallelMarkovitzRRRCpp(
  const arma::mat& returns,
  arma::mat& X0,
  const double lambda1,
  const double lambda2,
  const char penalty_type,
  const char step_size_type,
  const double step_size_constant,
  const unsigned int max_iter,
  const double tolerance
) {

  // if no initial `X0` is given, set it to hollow matrix with 1/N on the
  // off-diagonal
  if (X0.empty()) {

    X0 = arma::toeplitz(arma::join_cols(
      arma::vec::fixed<1>(arma::fill::zeros),
      arma::vec(returns.n_cols - 1, arma::fill::value(1. / returns.n_cols))
    ));

  }

  // initialize data
  MarkovitzRRRSolver solver(
    returns,
    X0,
    lambda1,
    lambda2,
    penalty_type,
    step_size_type,
    step_size_constant,
    max_iter,
    tolerance
  );

  // solve the optimization problem
  solver.Solve();

  // return output list
  return solver.GetOutputVector();

}


// ////////////////////////////////////////
// ////// ParallelMarkovitzRRRCpp /////////
//
// Rcpp::List ParallelMarkovitzRRRCpp(
//   const arma::mat& returns,
//   arma::mat& X0,
//   const double lambda1,
//   const arma::vec& lambda2_values,
//   const char penalty_type,
//   const char step_size_type,
//   const double step_size_constant,
//   const unsigned int max_iter,
//   const double tolerance
// ) {
//
//   // if no initial `X0` is given, set it to hollow matrix with 1/N on the
//   // off-diagonal
//   if (X0.empty()) {
//
//     X0 = arma::toeplitz(arma::join_cols(
//       arma::vec::fixed<1>(arma::fill::zeros),
//       arma::vec(returns.n_cols - 1, arma::fill::value(1. / returns.n_cols))
//     ));
//
//   }
//
//   // Determine the number of lambda2 values
//   const unsigned int n_lambda2 = lambda2_values.n_elem;
//
//   // Initialize matrices and vectors to store the results
//   arma::mat weights(returns.n_cols, n_lambda2);
//   arma::vec is_improved(n_lambda2);
//   arma::vec is_converged(n_lambda2);
//
//   // Create an instance of the parallel worker
//   ParallelMarkovitzRRRSolver worker(
//     weights,
//     is_improved,
//     is_converged,
//     returns,
//     X0,
//     lambda1,
//     lambda2_values,
//     penalty_type,
//     step_size_type,
//     step_size_constant,
//     max_iter,
//     tolerance
//   );
//
//   // Perform the parallel computation
//   RcppParallel::parallelFor(0, n_lambda2, worker);
//
//   // Return a list containing the optimal portfolio weights and solver status
//   // information
//   return Rcpp::List::create(
//     Rcpp::Named("weights") = weights,
//     Rcpp::Named("is_improved") = is_improved,
//     Rcpp::Named("is_converged") = is_converged
//   );
//
// }

