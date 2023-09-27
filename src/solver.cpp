// Author: Alberto Quaini

#include "solver.h"

//// Implementation of MarkovitzRRRSolver

// class constructor
MarkovitzRRRSolver::MarkovitzRRRSolver(
  const arma::mat& R,
  const arma::mat& X0,
  const double lambda,
  const char penalty_type,
  const char step_size_type,
  const double step_size_constant,
  const unsigned int max_iter
  // const double tolerance
) : // initialize object members at class construction
  R(R),
  T(R.n_rows),
  N(R.n_cols),
  X0(X0),
  X1(X0),
  Xbest(X0),
  lambda(lambda),
  objective(arma::vec(max_iter)),
  subgradient(X0),
  step_size_constant(step_size_constant),
  max_iter(max_iter),
  iter(1)
  // set ComputeStepSize function depending on the step_size_type
  // If step_size_type = 'c' for constant, then the returned function computes a
  // constant step size.
  // If step_size_type = 's' for square summable, then the returned function
  // computes a square summable but not summable step size.
  // Otherwise the function computes a not summable vanishing step size.
  // Default is step_size_type = 's'.
  // ComputeStepSize(
  //   [this, step_size_type](const unsigned int iter) -> double {
  //     switch (step_size_type) {
  //     case 'c': // Constant step size
  //       return this->step_size_constant;
  //     case 's': // Square summable not summable
  //       return this->step_size_constant / (iter + 1);
  //     case 'p': // (modified) Polyak
  //       {
  //         return this->step_size_constant / arma::sum(arma::sum(arma::square(
  //             this->subgradient
  //         )));
  //       }
  //     default:  // Not summable vanishing
  //       return this->step_size_constant / std::sqrt(iter + 1);
  //     }
  //   }
  // ),
  // set tolerance to the square root of the machine epsilon for double
  // tolerance(tolerance), // std::numeric_limits<double>::epsilon()
  // initialize status as "unsolved"
  // status("unsolved"),
{

  // compute the objective function at `X0`
  objective(0) = ComputeObjective(X0);

  // set `ComputeStepSize` according to `step_size_type`:
  // if `step_size_type` = 'c' for constant, then the returned function computes a
  // constant step size.
  // if `step_size_type` = 's' for square summable, then the returned function
  // computes a square summable but not summable step size.
  // otherwise the function computes a not summable vanishing step size.
  // default is `step_size_type` = 's'.
  switch(step_size_type) {

  case 'c': // Constant step size
    step_size = step_size_constant;
    // ComputeStepSize does nothing
    ComputeStepSize = [&]() -> void {};
    break;

  case 's': // Square summable not summable
    ComputeStepSize = [&]() -> void {
      step_size = step_size_constant / (iter + 1);
    };
    break;

  case 'p': // (modified) Polyak
    ComputeStepSize = [&]() -> void {
      step_size = step_size_constant / arma::sum(arma::sum(arma::square(
          subgradient
      )));
    };
    break;

  default:  // Not summable vanishing
    ComputeStepSize = [&]() -> void {
      step_size = step_size_constant / std::sqrt(iter + 1);
    };
    break;

  }

  // set `ComputeSubgradient` according to `penalty_type`:
  // if `penalty_type` = 'a' for alternative, then the subgradient accounts for
  // the penalty `lambda ||X||_*`. Otherwise, as default, it accounts for the
  // penalty `lambda ||R * X||_*`, for which there are two implementations,
  // one computing each time `svd(R * X)=USV'`, and the other one computing once
  // `svd(R)=USV'` and each time computing `qr(X'V)=QA` and `svd(AS)`.
  // the latter option is desirable when T>>N.
  switch(penalty_type) {

  case 'a': // alternative penalty: lambda ||X||_*
    ComputeSubgradient = [&]() -> void {
      Rcpp::Rcout << "ciao\n";

      // compute svd(X)
      arma::svd(U, sv, V, X0);

      // element in the subgradient of
      // 0.5 ||R - RX||_F^2 + lambda ||X||_*
      // with respect to X
      subgradient = lambda * U * V.t() + R.t() * (R * X0 - R);

    };
    break;

  default:
    // if N/T >= .9, then compute subgradient according to svd(R * X)
    if ((double)N/T >= .9) {

      ComputeSubgradient = [&]() -> void {
        Rcpp::Rcout << "hola\n";

        // compute R * X0
        const arma::mat RX0 = R * X0;

        // compute svd(R * X)
        arma::svd(U, sv, V, RX0);

        Rcpp::Rcout << "hey1\n";
        // element in the subgradient of
        // 0.5 ||R - RX||_F^2 + lambda ||R * X||_*
        // with respect to X
        subgradient = lambda * U.cols(0, N-1) * V.t() + R.t() * (R * X0 - R);
        Rcpp::Rcout << "you1\n";

      };
    // otherwise compute subgradient according to svd(R), qr(X'Vr) and svd(ASr)
    } else {
      // compute svd(R) once
      arma::svd(U, sv, V, R);
      // remove the last (T - N) columns from U
      U.shed_cols(N, T-1);

      ComputeSubgradient = [&]() -> void {
        Rcpp::Rcout << "salut\n";

        // compute qr(X'V)
        arma::mat Q, A;
        arma::qr(Q, A, X0.t() * V);

        // compute svd(A * S)
        arma::mat U1, V1;
        arma::vec sv1;

        arma::svd(U1, sv1, V1, A * arma::diagmat(sv));

        // element in the subgradient of
        // 0.5 ||R - RX||_F^2 + lambda ||R * X||_*
        // with respect to X
        subgradient = lambda * (U * V1)  * (Q * U1).t() + R.t() * (R * X0 - R);

      };

    }
    break;

  }

};

// compute the projected subgradient optimization path
void MarkovitzRRRSolver::Solve() {

  // main loop
  while(iter < max_iter) {

    // Compute the projected subgradient step based on the current iteration
    ProjectedSubgradientStep(iter);

    // update `X0` to `X1`
    X0 = X1;

    // increment `iter`
    ++iter;

  }

  // not to waste the last new `X0`, finalize with one more step
  ProjectedSubgradientStep(iter);

}

// Compute one projected subgradient step based on the current iteration
void MarkovitzRRRSolver::ProjectedSubgradientStep(const unsigned int iter) {

  Rcpp::Rcout << "hey\n";
  // compute the subgradient, stored in `this->subgradient`
  ComputeSubgradient();
  Rcpp::Rcout << "you\n";

  // compute the step size, stored in `this->step_size`
  ComputeStepSize();
  Rcpp::Rcout << "out\n";

  // update `X1`, remember that `X1` = `X0` before this computation
  X1 -= step_size * subgradient;
  Rcpp::Rcout << "here\n";

  // project `X1` on the space of hollow matrices
  X1.diag().zeros();

  // store objective function at `X1`
  objective(iter) = ComputeObjective(X1);

  // replace `Xbest` with `X1` if f(X1) <= f(Xbest)
  if (objective(iter) <= ComputeObjective(Xbest)) Xbest = X1;

  // // if the difference of subsequent solutions is smaller than the tolerance
  // // set the solver `status` to "solved"
  // if (arma::sum(arma::sum(arma::square(X1 - X0))) / (N * N) < tolerance) {
  //   status = "solved";
  // }

};

// compute objective value for given X
double MarkovitzRRRSolver::ComputeObjective(const arma::mat& X) const {

  // store R * X
  const arma::mat RX = R * X;

  // return the objective function at X
  // 0.5 ||R - R * X||_F^2 + lambda ||R * X||
  return 0.5 * arma::sum(arma::sum(arma::square(R - RX))) +
    lambda * arma::sum(arma::svd(RX));

};

// set lambda
void MarkovitzRRRSolver::SetLambda(const double lambda) {
  this->lambda = lambda;
};

// set initial point to hollow matrix with 1/N on the off-diagonal
void MarkovitzRRRSolver::SetX0ToHollow1OverN () {
  X0 = arma::toeplitz(arma::join_cols(
    arma::vec::fixed<1>(arma::fill::zeros),
    arma::vec(R.n_cols-1, arma::fill::value(1./R.n_cols))
  ));
};

// get objective
const arma::vec& MarkovitzRRRSolver::GetObjective() const {
  return objective;
};

// // get solver status
// const std::string& MarkovitzRRRSolver::GetStatus() const {
//   return status;
// };

// get solution
const arma::mat& MarkovitzRRRSolver::GetSolution() const {
  return Xbest;
};

// compute the optimal portfolio weights
arma::rowvec MarkovitzRRRSolver::GetOptimalPortfolioWeights() const {

  // compute the marginal variances of the residuals between returns R and
  // hedging returns R * X
  const arma::rowvec residuals_variance = arma::var(R - R * Xbest, 0, 0);

  // compute the unscaled optimal weights: Sigma^(-1)'1 where Sigma^(-1),
  // the inverse variance covariance matrix of returns, is computed via
  // the solution X and the marginal variances of the residuals
  const arma::rowvec weights = arma::sum(
    arma::diagmat(residuals_variance) * (arma::eye(N, N) - Xbest)
  );

  return weights / arma::sum(weights);

};
