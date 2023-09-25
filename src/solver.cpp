// Author: Alberto Quaini

#include "solver.h"

//// Implementation of MarkovitzRRRSolver

// class constructor
MarkovitzRRRSolver::MarkovitzRRRSolver(
  const arma::mat& R,
  const arma::mat& X0,
  const double lambda,
  const char step_size_type,
  const double step_size_constant,
  const double tolerance
) : // initialize object members at class construction
  R(R),
  T(R.n_rows),
  N(R.n_cols),
  X0(X0),
  X1(X0),
  Xbest(X0),
  lambda(lambda),
  subgradient(X0),
  // set tolerance to the square root of the machine epsilon for double
  tolerance(tolerance), // std::numeric_limits<double>::epsilon()
  // initialize status as "unsolved"
  status("unsolved"),
  step_size_constant(step_size_constant),
  // set ComputeStepSize function depending on the step_size_type
  // If step_size_type = 'c' for constant, then the returned function computes a
  // constant step size.
  // If step_size_type = 's' for square summable, then the returned function
  // computes a square summable but not summable step size.
  // Otherwise the function computes a not summable vanishing step size.
  // Default is step_size_type = 's'.
  ComputeStepSize(
    [this, step_size_type](const unsigned int iter) -> double {
      switch (step_size_type) {
      case 'c': // Constant step size
        return this->step_size_constant;
      case 's': // Square summable not summable
        return this->step_size_constant / (iter + 1);
      case 'p': // (modified) Polyak
        {
          return this->step_size_constant / arma::sum(arma::sum(arma::square(
              this->subgradient
          )));
        }
      default:  // Not summable vanishing
        return this->step_size_constant / std::sqrt(iter + 1);
      }
    }
  )
{};


// Compute the projected subgradient step based on the current iteration
void MarkovitzRRRSolver::ProjectedSubgradientStep(const unsigned int iter) {

  // compute the subgradient, stored in this->subgradient
  ComputeSubgradient();

  // compute step: X1 = X0 - step_size * G0
  // where G0 is an element in the subgradient of the objective function at X0
  // remember that X1 = X0 before this computation
  X1 -= ComputeStepSize(iter) * subgradient;

  // project X1 on the space of hollow matrices
  X1.diag().zeros();

  // store objective function
  objective.push_back(ComputeObjective(X1));

  // replace Xbest with X1 if f(X1) <= f(Xbest)
  if (objective.back() <= ComputeObjective(Xbest)) Xbest = X1;

  // if the difference of subsequent solutions is smaller than the tolerance
  // set the solver status to "solved"
  if (arma::sum(arma::sum(arma::square(X1 - X0))) / (N * N) < tolerance) {
    status = "solved";
  }

  // set X0 to X1
  X0 = X1;

};

// compute the subgradient
void MarkovitzRRRSolver::ComputeSubgradient() {

  const arma::mat RX0 = R * X0;

  // SVD of RX0
  arma::mat U(T, N);
  arma::mat V(N, N);
  arma::vec sv(N);

  arma::svd(U, sv, V, RX0);

  // element in the subgradient of
  // 0.5 ||R - RX||_F^2 + lambda ||RX||_*
  // with respect to X
  subgradient = R.t() * (-R + RX0 + lambda * U.cols(0, N-1) * V.t());

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
const std::vector<double>& MarkovitzRRRSolver::GetObjective() const {
  return objective;
};

// get solver status
std::string MarkovitzRRRSolver::GetStatus() const {
  return status;
};

// get solution
arma::mat MarkovitzRRRSolver::GetSolution() const {
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
