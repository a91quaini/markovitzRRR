// // Author: Alberto Quaini
//
// #include "parallel_markovitz_rrr_solver.h"
// #include "markovitz_rrr_solver.h"
//
// //// Implementation of ParallelMarkovitzRRRSolver
//
// // Class constructor
// ParallelMarkovitzRRRSolver::ParallelMarkovitzRRRSolver(
//   arma::mat& weights,
//   arma::vec& is_improved,
//   arma::vec& is_converged,
//   const arma::mat& returns,
//   arma::mat& X0,
//   const double lambda1,
//   const arma::vec& lambda2_values,
//   const char penalty_type,
//   const char step_size_type,
//   const double step_size_constant,
//   const unsigned int max_iter,
//   const double tolerance
// ) :
//   weights(weights),
//   is_improved(is_improved),
//   is_converged(is_converged),
//   returns(returns),
//   X0(X0),
//   lambda1(lambda1),
//   lambda2_values(lambda2_values),
//   penalty_type(penalty_type),
//   step_size_type(step_size_type),
//   step_size_constant(step_size_constant),
//   max_iter(max_iter),
//   tolerance(tolerance) {};
//
// // Operator function to perform parallel computation
// void ParallelMarkovitzRRRSolver::operator()(std::size_t begin, std::size_t end) {
//
//   // Loop over the range provided for parallel execution
//   for (std::size_t idx = begin; idx < end; idx++) {
//
//     // Get the current lambda2 value from the vector of lambda2 values
//     double lambda2 = lambda2_values[idx];
//
//     // Instantiate the MarkovitzRRRSolver with the current lambda2 value
//     MarkovitzRRRSolver solver(
//       returns,
//       X0,
//       lambda1,
//       lambda2,
//       penalty_type,
//       step_size_type,
//       step_size_constant,
//       max_iter,
//       tolerance
//     );
//
//     // Run the optimization solver
//     solver.Solve();
//
//     // Store the optimal portfolio weights and solver status information
//     weights.row(idx) = solver.GetWeights();
//     is_improved(idx) = solver.GetIsImproved();
//     is_converged(idx) = solver.GetIsConverged();
//
//   }
//
// }
//
