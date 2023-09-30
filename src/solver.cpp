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
  const unsigned int max_iter,
  const double tolerance
) : // initialize object members at class construction
  R(R),
  T(R.n_rows),
  N(R.n_cols),
  X0(X0),
  X1(X0),
  Xbest(X0),
  lambda(lambda),
  objective(arma::vec(max_iter + 1)),
  ComputeSubgradient(SetSubgradientFunction(penalty_type)),
  step_size_constant(step_size_constant),
  ComputeStepSize(SetStepSizeFunction(step_size_type)),
  max_iter(max_iter),
  iter(0),
  tolerance(tolerance)
{

  // compute the objective function at `X0`
  objective(iter) = ComputeObjective(X0);

  // if N is small compared to T, the `ComputeSubgradient` function assumes
  // that the svd of R is computed and stored in U, sv and V
  if ((double)N/T < .9) {

    // compute svd(R) once
    arma::svd(U, sv, V, R);
    // remove the last (T - N) columns from U
    U.shed_cols(N, T-1);

  }

};

// compute the projected subgradient optimization path
void MarkovitzRRRSolver::Solve() {

  // main loop with solution check
  if (tolerance > 0) {

    while(++iter < max_iter) {

      Rcpp::Rcout << iter << "\n";

      // Compute the projected subgradient step based on the current iteration
      ComputeProjectedSubgradientStep(iter);

      // if ||X1 - X0||_F^2 < tolerance quit loop
      if (arma::accu(arma::square(X1 - X0)) < tolerance) {

        // remove elements
        objective.shed_rows(iter + 1, max_iter);
        break;

      }

      // update `X0` to `X1`
      X0 = X1;

    }

  // main loop without solution check
  } else {

    // main loop
    while(++iter < max_iter) {

      // Compute the projected subgradient step based on the current iteration
      ComputeProjectedSubgradientStep(iter);

      // update `X0` to `X1`
      X0 = X1;

    }

    // not to waste the last new `X0`, finalize with one more step
    ComputeProjectedSubgradientStep(iter);

  }

}

// Compute one projected subgradient step based on the current iteration
void MarkovitzRRRSolver::ComputeProjectedSubgradientStep(const unsigned int iter) {

  // compute the subgradient, stored in `this->subgradient`
  ComputeSubgradient();

  // update `X1`, remember that `X1` = `X0` before this computation
  X1 -= ComputeStepSize() * subgradient;

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


//////////////////////
//// Step Size ///////

// compute constant step_size
double MarkovitzRRRSolver::ComputeStepSizeConstant() const {

  return step_size_constant;

};

// compute step_size for constant step length
double MarkovitzRRRSolver::ComputeStepSizeConstantStepLength() const {

  return step_size_constant / arma::norm(subgradient, "fro");

};

// compute not summable vanishing step_size
double MarkovitzRRRSolver::ComputeStepSizeNotSummableVanishing() const {

  return step_size_constant / std::sqrt(iter + 1);

  };

// compute square summable not summable step_size
double MarkovitzRRRSolver::ComputeStepSizeSquareSummableNotSummable() const {

  return step_size_constant / (iter + 1);

};

// compute square summable not summable step_size
double MarkovitzRRRSolver::ComputeStepSizeModifiedPolyak() const {

  return (step_size_constant + objective(iter - 1) -
    arma::min(objective.head(iter))) / arma::norm(subgradient, "fro");

};

////////////////////////
//// Subgradient ///////

// Compute subgradient for large N
void MarkovitzRRRSolver::ComputeSubgradientForLargeN() {

  // compute R * X0
  const arma::mat RX0 = R * X0;

  // compute svd(R * X)
  arma::svd(U, sv, V, RX0);

  // element in the subgradient of
  // 0.5 ||R - RX||_F^2 + lambda ||R * X||_*
  // with respect to X
  subgradient = lambda * R.t() * U.cols(0, N-1) * V.t() +
    R.t() * (R * X0 - R);

}

// Compute subgradient for small N
void MarkovitzRRRSolver::ComputeSubgradientForSmallN() {

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
  subgradient = lambda * R.t() * (U * V1)  * (Q * U1).t() +
    R.t() * (R * X0 - R);

}

// Compute alternative subgradient
void MarkovitzRRRSolver::ComputeSubgradientAlternative() {

  // compute svd(X)
  arma::svd(U, sv, V, X0);

  // element in the subgradient of
  // 0.5 ||R - RX||_F^2 + lambda ||X||_*
  // with respect to X
  subgradient = lambda * U * V.t() + R.t() * (R * X0 - R);

}

// compute the optimal portfolio weights
arma::rowvec MarkovitzRRRSolver::ComputeOptimalPortfolioWeights() {

  // compute the marginal variances of the residuals between returns R and
  // hedging returns R * X
  const arma::rowvec residuals_variance = arma::var(R - R * Xbest, 0, 0);

  // compute the unscaled optimal weights: Sigma^(-1)'1 where Sigma^(-1),
  // the inverse variance covariance matrix of returns, is computed via
  // the solution X and the marginal variances of the residuals
  const arma::rowvec weights = sum(
    arma::diagmat(residuals_variance) * (arma::eye(N, N) - Xbest)
  );

  return weights / arma::sum(weights);

}

///////////////
/// setters ///

// set function `ComputeStepSize` according to `step_size_type`:
// if `step_size_type` = 'c' for constant, then the returned function computes a
// constant step size: `step_size = step_size_constant`.
// if `step_size_type` = 'l' for length, then the returned function computes a
// step size that keeps a constant step length:
// `step_size = step_size_constant / ||subgradient||_F`.
// if `step_size_type` = 's' for square summable, then the returned function
// computes a square summable but not summable step size:
// `step_size = step_size_constant / (iter + 1)`.
// otherwise the function computes a not summable vanishing step size:
// `step_size = step_size_constant / sqrt(iter + 1)`.
// default is `step_size_type` = 'd'.
std::function<double(void)> MarkovitzRRRSolver::SetStepSizeFunction(
    const char step_size_type
) const {

  switch (step_size_type) {

  case 'c':
    return std::bind(&MarkovitzRRRSolver::ComputeStepSizeConstant, this);

  case 'l':
    return std::bind(&MarkovitzRRRSolver::ComputeStepSizeConstantStepLength, this);

  case 's':
    return std::bind(&MarkovitzRRRSolver::ComputeStepSizeSquareSummableNotSummable, this);

  case 'p':
    return std::bind(&MarkovitzRRRSolver::ComputeStepSizeModifiedPolyak, this);

  default:
    return std::bind(&MarkovitzRRRSolver::ComputeStepSizeNotSummableVanishing, this);

  }

};

// set `ComputeSubgradient` according to `penalty_type`:
// if `penalty_type` = 'a' for alternative, then the subgradient accounts for
// the penalty `lambda ||X||_*`. Otherwise, as default, it accounts for the
// penalty `lambda ||R * X||_*`, for which there are two implementations,
// one computing each time `svd(R * X)=USV'`, and the other one computing once
// `svd(R)=USV'` and each time computing `qr(X'V)=QA` and `svd(AS)`.
// the latter option is desirable when T>>N.
std::function<void(void)> MarkovitzRRRSolver::SetSubgradientFunction(
  const char penalty_type
) {

  switch (penalty_type) {

  case 'a':
    return std::bind(&MarkovitzRRRSolver::ComputeSubgradientAlternative, this);

  default:
    return (double)N/T >= .9 ?
      std::bind(&MarkovitzRRRSolver::ComputeSubgradientForLargeN, this) :
      std::bind(&MarkovitzRRRSolver::ComputeSubgradientForSmallN, this);

  }

}

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

///////////////
/// getters ///

// get objective
const arma::vec& MarkovitzRRRSolver::GetObjective() const {
  return objective;
};

// get solution
const arma::mat& MarkovitzRRRSolver::GetSolution() const {
  return Xbest;
};
