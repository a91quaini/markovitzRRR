// // Author: Alberto Quaini
//
// #ifndef PARALLEL_MARKOVITZ_RRR_SOLVER_H
// #define PARALLEL_MARKOVITZ_RRR_SOLVER_H
//
// #include <RcppArmadillo.h>
// // [[Rcpp::depends(RcppParallel)]]
// #include <RcppParallel.h>
//
// struct ParallelMarkovitzRRRSolver : public RcppParallel::Worker {
//
//   // Output to store the results
//   arma::mat& weights;
//   arma::vec& is_improved;
//   arma::vec& is_converged;
//
//   // Inputs
//   const arma::mat& returns;
//   arma::mat X0;
//   const double lambda1;
//   const arma::vec& lambda2_values;
//   const char penalty_type;
//   const char step_size_type;
//   const double step_size_constant;
//   const unsigned int max_iter;
//   const double tolerance;
//
//   // Constructor
//   ParallelMarkovitzRRRSolver(
//     arma::mat& weights,
//     arma::vec& is_improved,
//     arma::vec& is_converged,
//     const arma::mat& returns,
//     arma::mat& X0,
//     const double lambda1 = 0.,
//     const arma::vec& lambda2_values = arma::zeros(1),
//     const char penalty_type = 'd',
//     const char step_size_type = 'd',
//     const double step_size_constant = 0.,
//     const unsigned int max_iter = 10000,
//     const double tolerance = 0.
//   );
//
//   // Parallel operator
//   void operator()(std::size_t begin, std::size_t end);
//
// };
//
// #endif
