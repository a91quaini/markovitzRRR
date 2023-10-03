// Author: Alberto Quaini

#include "solver.h"

//// Implementation of MarkovitzRRRSolver

// class constructor
MarkovitzRRRSolver::MarkovitzRRRSolver(
  const arma::mat& R,
  const arma::mat& X0,
  const double lambda,
  const char objective_type,
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
  objective(arma::vec(max_iter)),
  ComputeObjective(SetObjectiveFunction(objective_type)),
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

      // Compute the projected subgradient step based on the current iteration
      ComputeProjectedSubgradientStep(iter);

      // if ||X1 - X0||_F^2 < tolerance quit loop
      if (arma::accu(arma::square(X1 - X0)) < tolerance) {

        // remove elements
        objective = objective.head(iter);
        // objective.shed_rows(iter + 1, max_iter);
        break;

      }

      // update `X0` to `X1`
      X0 = X1;

    }

  // main loop without solution check
  } else {

    // main loop
    while(++iter < max_iter - 1) {

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

  // update `X1`, i.e., `X1 = X0 - step_size * subgradient`
  // remember that `X1` = `X0` before this computation
  X1 -= ComputeStepSize() * subgradient;

  // project `X1` on the space of hollow matrices --> zero out the diagonal
  X1.diag().zeros();

  // store objective function at `X1`
  objective(iter) = ComputeObjective(X1);

  // replace `Xbest` with `X1` if `f(X1) <= f(Xbest)`
  if (objective(iter) <= ComputeObjective(Xbest)) Xbest = X1;

};

// compute objective value for given X
double MarkovitzRRRSolver::ComputeDefaultObjective(const arma::mat& X) const {

  // store R * X
  const arma::mat RX = R * X;

  // return the objective function at X
  // 1/2 ||R - RX||_F^2 + lambda ||RX||
  return .5 * arma::accu(arma::square(R - RX)) +
    lambda * arma::accu(arma::svd(RX));

};

// compute objective value for given X
double MarkovitzRRRSolver::ComputeAlternativeObjective(const arma::mat& X) const {

  // return the objective function at X
  // 1/2 ||R - RX||_F^2 + lambda ||RX||
  return .5 * arma::accu(arma::square(R - R * X)) +
    lambda * arma::accu(arma::svd(X));

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
  const arma::rowvec weights = arma::sum(
    arma::diagmat(residuals_variance) * (arma::eye(N, N) - Xbest), 0
  );

  return weights / arma::accu(weights);

}

///////////////
/// setters ///

// set `ComputeObjective` according to `objective_type`:
// `'d'` for default, i.e., objective given by
// `.5||R - RX||_F^2 + lambda * ||RX||_*`;
// `'a'` for alternative, i.e., objective given by
// `.5||R - RX||_F^2 + lambda * ||X||_*`. Default is `'d'`.
std::function<double(const arma::mat&)> MarkovitzRRRSolver::SetObjectiveFunction(
  const char objective_type
) const {

  switch (objective_type) {

  case 'd':
    // return [this](const arma::mat& X) -> double { return ComputeDefaultObjective(X); };
    return std::bind(
      &MarkovitzRRRSolver::ComputeDefaultObjective,
      this,
      std::placeholders::_1
    );

  default:
    // return [this](const arma::mat& X) -> double { return ComputeAlternativeObjective(X); };
    return std::bind(
      &MarkovitzRRRSolver::ComputeAlternativeObjective,
      this,
      std::placeholders::_1
    );

  }

}

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

  case 'd':
    return (double)N/T >= .9 ?
      std::bind(&MarkovitzRRRSolver::ComputeSubgradientForLargeN, this) :
      std::bind(&MarkovitzRRRSolver::ComputeSubgradientForSmallN, this);

  default:
    return std::bind(&MarkovitzRRRSolver::ComputeSubgradientAlternative, this);

  }

}

// set function `ComputeStepSize` according to `step_size_type`:
// if `step_size_type` = 'd' for default, the function computes a not summable
// vanishing step size: `step_size = step_size_constant / sqrt(iter + 1)`.
// if `step_size_type` = 's' for square summable, then the returned function
// computes a square summable but not summable step size:
// `step_size = step_size_constant / (iter + 1)`.
// if `step_size_type` = 'l' for length, then the returned function computes a
// step size that keeps a constant step length:
// `step_size = step_size_constant / ||subgradient||_F`.
// if `step_size_type` = 'p' for Polyak's, then the returned function computes a
// modified Polyak's step size:
// `step_size = (step_size_constant - objective_iter + min{objective_i : i=1,..,iter}) / ||subgradient||_F`.
// if `step_size_type` = 'c' for constant, then the returned function computes a
// constant step size: `step_size = step_size_constant`.
// default is `step_size_type` = 'd'.
std::function<double(void)> MarkovitzRRRSolver::SetStepSizeFunction(
    const char step_size_type
) const {

  switch (step_size_type) {

  case 'd':
    return std::bind(&MarkovitzRRRSolver::ComputeStepSizeNotSummableVanishing, this);

  case 's':
    return std::bind(&MarkovitzRRRSolver::ComputeStepSizeSquareSummableNotSummable, this);

  case 'l':
    return std::bind(&MarkovitzRRRSolver::ComputeStepSizeConstantStepLength, this);

  case 'p':
    return std::bind(&MarkovitzRRRSolver::ComputeStepSizeModifiedPolyak, this);

  default:
    return std::bind(&MarkovitzRRRSolver::ComputeStepSizeConstant, this);

  }

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
