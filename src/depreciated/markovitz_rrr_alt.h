// Rcpp::List MarkovitzRRRAltCpp(
//   const arma::mat& R,
//   const double tau,
//   const double lambda = 0.,
//   const unsigned int max_iter = 10000,
//   const double tolerance = -1.
// );

// // ////////////////////////////////
// // ////// MarkovitzRRRAltCpp //////
// //
// // Rcpp::List MarkovitzRRRAltCpp(
// //   const arma::mat& R,
// //   const double tau,
// //   const double lambda,
// //   const unsigned int max_iter,
// //   const double tolerance
// // ) {
// //
// //   // initialize data
// //   DykstraAP solver(
// //     R,
// //     tau,
// //     lambda,
// //     max_iter,
// //     tolerance
// //   );
// //
// //   // solve the optimization problem
// //   solver.Solve();
// //
// //   return Rcpp::List::create(
// //     Rcpp::Named("solution") = solver.GetSolution(),
// //     Rcpp::Named("objective") = solver.GetObjective()
// //     // Rcpp::Named("weights") = solver.ComputeOptimalPortfolioWeights()
// //     // Rcpp::Named("status") = solver.GetStatus(),
// //   );
// //
// // }
