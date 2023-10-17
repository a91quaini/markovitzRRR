// Author: Alberto Quaini

#include "solver.h"

//// Implementation of MarkovitzRRRSolver

// class constructor
MarkovitzRRRSolver::MarkovitzRRRSolver(
  const arma::mat& R,
  const arma::mat& X0,
  const double lambda1,
  const double lambda2,
  const char penalty_type,
  const char step_size_type,
  const double step_size_constant,
  const unsigned int max_iter,
  const double tolerance
) : // initialize object members at class construction
  R(R),
  T(R.n_rows),
  N(R.n_cols),
  minNT(std::min(N, T)),
  X(X0),
  Xbest(X0),
  lambda1(lambda1),
  lambda2(lambda2),
  max_iter(max_iter),
  iter(1),
  ComputeObjective(SetObjectiveFunction(penalty_type, lambda1, lambda2)),
  objective(max_iter),
  ComputeSubgradient(SetSubgradientFunction(penalty_type, lambda1, lambda2)),
  step_size_constant(SetStepSizeConstant(step_size_constant, lambda2)),
  ComputeStepSize(SetStepSizeFunction(step_size_type)),
  tolerance(tolerance),
  // ancilliary
  X_norm(max_iter),
  Sigma_inv_norm(max_iter),
  Weights(N, max_iter)
{

  // if N is small compared to T, the `ComputeSubgradient` function assumes
  // that the svd of R is computed and stored in U, sv and V
  // note: here `default_NT_ratio = .7`
  if ((double)N/T < .7) {

    // compute svd(R) once
    arma::svd(U, sv, V, R);
    // remove the last (T - N) columns from U
    U.shed_cols(N, T-1);

  }

  // compute the objective function at `X0`
  objective(0) = ComputeObjective();

  // set the best objective value to `objective(0)`
  objective_best = objective(0);

};

// compute the projected subgradient optimization path
void MarkovitzRRRSolver::Solve() {

  // main loop with solution check
  if (tolerance > 0) {

    // main loop
    do {

      // set `X0` to `X`
      X0 = X;

      // Compute the projected subgradient step based on the current iteration
      ComputeProjectedSubgradientStep();

    } while(
        // while iter < max_iter - 1
        (++iter < max_iter) &&
        // and ||X - X0||_F^2 > tolerance
        (arma::accu(arma::square(X - X0)) > tolerance)
    );

    // remove elements in excess in `objective`
    objective = objective.head(iter);

  // main loop without solution check
  } else {

    // main loop
    do {

      // set `X0` to `X`
      X0 = X;

      // Compute the projected subgradient step based on the current iteration
      ComputeProjectedSubgradientStep();

      // ancilliary
      X_norm(iter) = arma::norm(X, "fro");

      // compute the inverse marginal variances of the residuals between returns
      // R and hedging returns R * X
      const arma::mat residuals_variance_inv = arma::diagmat(
        1. / arma::var(R - R * Xbest, 0, 0)
      );

      ////
      const arma::mat Sigma_inv = residuals_variance_inv -
        residuals_variance_inv * Xbest;
      Sigma_inv_norm(iter) = arma::norm(Sigma_inv, "fro");

      arma::vec wweights = arma::sum(Sigma_inv, 0).t();

      // project weights into the unit simplex
      wweights /= arma::accu(wweights);
      Weights.col(iter) = wweights;

    } while(++iter < max_iter);

  }

}

// solve the unpenalized Markovitz optimization problem
void MarkovitzRRRSolver::SolveUnpenalizedMarkovitz() {

  // compute `X` by running N linear regressions of R^(i), the i-th column of R,
  // on R^(-i), i.e., R without the i-th column
  for (unsigned int i = 0; i < N; ++i) {

    arma::mat Ri = R;
    Ri.shed_col(i);
    const arma::vec coeff = arma::solve(
      Ri.t() * Ri,
      Ri.t() * R.col(i),
      arma::solve_opts::likely_sympd
    );

    X.col(i) = arma::join_vert(
      coeff.head(i),
      arma::zeros(1),
      coeff.tail(N-1-i)
    );

  }

  // store the objective function value `1/2||R - R * X||_F^2`
  objective(1) = .5 * arma::accu(arma::square(R - R * X));

  // keep only the first two function evaluations
  objective = objective.head(2);

}

// Compute one projected subgradient step based on the current iteration
void MarkovitzRRRSolver::ComputeProjectedSubgradientStep() {

  // compute the subgradient, stored in `this->subgradient`
  ComputeSubgradient();

  // update `X`, i.e., `X = X0 - step_size * subgradient`
  // remember that `X` = `X0` before this computation
  X -= ComputeStepSize() * subgradient;

  // project `X` on the space of hollow matrices --> zero out the diagonal
  X.diag().zeros();

  // store objective function at `X`
  objective(iter) = ComputeObjective();

  // replace `Xbest` with `X` if `f(X) <= f(Xbest)` and update `objective_best`
  if (objective(iter) <= objective_best) {

    objective_best = objective(iter);
    Xbest = X;

  }

};

//// Objective function

// compute the objective value with Ridge penalty:
// `1/2 ||R - RX||_F^2 + lambda2/2||X||_F^2` for given `X`
double MarkovitzRRRSolver::ComputeMainObjectiveRidge() const {

  // Rcpp::Rcout << "ComputeMainObjectiveRidge\n";
  return .5 * arma::accu(arma::square(R - R * X)) +
    .5 * lambda2 * arma::accu(X % X);

};

// compute the objective function with Nuclear penalty:
// `1/2 ||R - RX||_F^2 + lambda1 ||RX||_*` for given `X`
double MarkovitzRRRSolver::ComputeMainObjectiveNuclear() const {

  // Rcpp::Rcout << "ComputeMainObjectiveNuclear\n";
  const arma::mat RX = R * X;
  return .5 * arma::accu(arma::square(R - RX)) +
    lambda1 * arma::accu(arma::svd(RX));

};

// compute the objective function with Nuclear penalty when N is "small":
// `1/2 ||R - RX||_F^2 + lambda1 ||RX||_*` for given `X`
double MarkovitzRRRSolver::ComputeMainObjectiveNuclearSmallN() const {

  // Rcpp::Rcout << "ComputeMainObjectiveNuclearSmallN\n";
  // compute `qr(X'V)`, where `V` contains the right singular vectors of `R`
  arma::mat Q, A;
  arma::qr(Q, A, X.t() * V);

  // return the svd decomposition of `AS`, where `S` is the diagonal matrix
  // of singular values of `R`
  return .5 * arma::accu(arma::square(R - R * X)) +
    lambda1 * arma::accu(arma::svd(A * arma::diagmat(sv)));

};

// compute the objective function with Nuclear and Ridfe penalties:
// `1/2 ||R - RX||_F^2 + lambda1 ||RX||_* + lambda2/2||X||_F^2` for given `X`
double MarkovitzRRRSolver::ComputeMainObjectiveNuclearRidge() const {

  // Rcpp::Rcout << "ComputeMainObjectiveNuclearRidge\n";
  const arma::mat RX = R * X;
  return .5 * arma::accu(arma::square(R - RX)) +
    lambda1 * arma::accu(arma::svd(RX)) +
    lambda2 * arma::accu(X % X);

};

// compute the objective function with Nuclear and Ridfe penalties:
// `1/2 ||R - RX||_F^2 + lambda1 ||RX||_* + lambda2/2||X||_F^2` for given `X`
// when N is "small"
double MarkovitzRRRSolver::ComputeMainObjectiveNuclearRidgeSmallN() const {

  // Rcpp::Rcout << "ComputeMainObjectiveNuclearRidgeSmallN\n";
  // compute `qr(X'V)`, where `V` contains the right singular vectors of `R`
  arma::mat Q, A;
  arma::qr(Q, A, X.t() * V);

  // return the svd decomposition of `AS`, where `S` is the diagonal matrix
  // of singular values of `R`
  return .5 * arma::accu(arma::square(R - R * X)) +
    lambda1 * arma::accu(arma::svd(A * arma::diagmat(sv))) +
    lambda2 * arma::accu(X % X);

};

// compute the objective function with the Nuclear penalty on `X`:
// `1/2 ||R - RX||_F^2 + lambda1 ||X||_*` for given `X`
double MarkovitzRRRSolver::ComputeMainObjectiveAlternativeNuclear() const {

  // Rcpp::Rcout << "ComputeMainObjectiveAlternativeNuclear\n";
  return .5 * arma::accu(arma::square(R - R * X)) +
    lambda1 * arma::accu(arma::svd(X));

};

// compute the objective function with the Nuclear and Ridge penalties on `X`:
// `1/2 ||R - RX||_F^2 + lambda1 ||X||_* + lambda2/2||X||_F^2` for given `X`
double MarkovitzRRRSolver::ComputeMainObjectiveAlternativeNuclearRidge() const {

  // Rcpp::Rcout << "ComputeMainObjectiveAlternativeNuclearRidge\n";
  return .5 * arma::accu(arma::square(R - R * X)) +
    lambda1 * arma::accu(arma::svd(X)) +
    lambda2 * arma::accu(X % X);

};

////////////////////////
//// Subgradient ///////

// compute the subgradient of the objective value with Ridge penalty:
// `lambda2 * X0 - R'R + R'R * X0`
void MarkovitzRRRSolver::ComputeSubgradientMainObjectiveRidge() {

  // Rcpp::Rcout << "ComputeSubgradientMainObjectiveRidge\n";
  const arma::mat RtR = R.t() * R;
  subgradient = lambda2 * X0 - RtR + RtR * X0;

};

// compute the subgradient of the objective value with Nuclear penalty:
// `lambda1 * R'UV' - R'R + R'R * X0`, where
// `svd(R * X0) = U * S * V'`
void MarkovitzRRRSolver::ComputeSubgradientMainObjectiveNuclear() {

  // Rcpp::Rcout << "ComputeSubgradientMainObjectiveNuclear\n";
  const arma::mat RtR = R.t() * R;

  arma::svd(U, sv, V, R * X0);

  subgradient = lambda1 * R.t() * U.head_cols(minNT) * V.head_cols(minNT).t() +
    RtR * X0 - RtR;

}

// compute the subgradient of the objective value with Nuclear penalty:
// `lambda1 * R'UV' - R'R + R'R * X0`, where
// `svd(R * X0) = U * S * V'`, for "small" N
void MarkovitzRRRSolver::ComputeSubgradientMainObjectiveNuclearSmallN() {

  // Rcpp::Rcout << "ComputeSubgradientMainObjectiveNuclearSmallN\n";
  const arma::mat RtR = R.t() * R;

  // compute qr(X'V)
  arma::mat Q, A;
  arma::qr(Q, A, X0.t() * V);

  // compute svd(A * S)
  arma::mat U1, V1;
  arma::vec sv1;

  arma::svd(U1, sv1, V1, A * arma::diagmat(sv));

  subgradient = lambda1 * R.t() * (U * V1) * (Q * U1).t() +
    RtR * X0 - RtR;

}

// compute the subgradient of the objective value with Nuclear and Ridge penalty:
// `lambda1 * R'UV' + lambda2 * X0 - R'R + R'R * X0`, where
// `svd(R * X0) = U * S * V'`
void MarkovitzRRRSolver::ComputeSubgradientMainObjectiveNuclearRidge() {

  // Rcpp::Rcout << "ComputeSubgradientMainObjectiveNuclearRidge\n";
  const arma::mat RtR = R.t() * R;

  arma::svd(U, sv, V, R * X0);

  subgradient = lambda1 * R.t() * U.head_cols(minNT) * V.head_cols(minNT).t() +
    lambda2 * X0 + RtR * X0 - RtR;

}

// compute the subgradient of the objective value with Nuclear and Ridge penalty:
// `lambda1 * R'UV' + lambda2 * X0 - R'R + R'R * X0`, where
// `svd(R * X0) = U * S * V'`, for N "small"
void MarkovitzRRRSolver::ComputeSubgradientMainObjectiveNuclearRidgeSmallN() {

  // Rcpp::Rcout << "ComputeSubgradientMainObjectiveNuclearRidgeSmallN\n";
  const arma::mat RtR = R.t() * R;

  // compute qr(X'V)
  arma::mat Q, A;
  arma::qr(Q, A, X0.t() * V);

  // compute svd(A * S)
  arma::mat U1, V1;
  arma::vec sv1;

  arma::svd(U1, sv1, V1, A * arma::diagmat(sv));

  subgradient = lambda1 * R.t() * (U * V1) * (Q * U1).t() +
    lambda2 * X0 + RtR * X0 - RtR;

}

// compute the subgradient of the objective value with the alternative Nuclear
// penalty:
// `lambda1 * U * V' - R'R + R'R * X0`, where
// `svd(X0) = U * S * V'`
void MarkovitzRRRSolver::ComputeSubgradientMainObjectiveAlternativeNuclear() {

  // Rcpp::Rcout << "ComputeSubgradientMainObjectiveAlternativeNuclear\n";
  const arma::mat RtR = R.t() * R;

  arma::svd(U, sv, V, X0);

  subgradient = lambda1 * U * V.t() + RtR * X0 - RtR;

}

// compute the subgradient of the objective value with the alternative Nuclear
// and the Ridge penalty:
// `lambda1 * U * V' + lambda2 * X - R'R + R'R * X0`, where
// `svd(X0) = U * S * V'`
void MarkovitzRRRSolver::ComputeSubgradientMainObjectiveAlternativeNuclearRidge() {

  // Rcpp::Rcout << "ComputeSubgradientMainObjectiveAlternativeNuclearRidge\n";
  const arma::mat RtR = R.t() * R;

  arma::svd(U, sv, V, X0);

  subgradient = lambda1 * U * V.t() + lambda2 * X + RtR * X0 - RtR;

}

//////////////////////
//// Step Size ///////

// compute constant step_size
double MarkovitzRRRSolver::ComputeStepSizeConstant() const {

  return step_size_constant;

};

// compute not summable vanishing step_size
double MarkovitzRRRSolver::ComputeStepSizeNotSummableVanishing() const {

  return step_size_constant / std::sqrt(iter + 1);

};

// compute square summable not summable step_size
double MarkovitzRRRSolver::ComputeStepSizeSquareSummableNotSummable() const {

  return step_size_constant / (iter + 1);

};

// compute step_size for constant step length
double MarkovitzRRRSolver::ComputeStepSizeConstantStepLength() const {

  return step_size_constant / arma::norm(subgradient, "fro");

};

// compute square summable not summable step_size
double MarkovitzRRRSolver::ComputeStepSizeModifiedPolyak() const {

  return (step_size_constant + objective(iter - 1) -
          arma::min(objective.head(iter))) / arma::norm(subgradient, "fro");

};

// compute the optimal portfolio weights
void MarkovitzRRRSolver::ComputeOptimalPortfolioWeights() {

  // compute the inverse marginal variances of the residuals between returns
  // R and hedging returns R * X
  const arma::mat residuals_variance_inv = arma::diagmat(
    1. / arma::var(R - R * Xbest, 0, 0)
  );

  // compute the unscaled optimal weights: Sigma^(-1)'1 where Sigma^(-1),
  // the inverse variance covariance matrix of returns, is computed via
  // the solution X and the marginal variances of the residuals
  weights = arma::sum(
    residuals_variance_inv - residuals_variance_inv * Xbest, 0
  );

  // project weights into the unit simplex
  weights /= arma::accu(weights);

};

///////////////
/// setters ///

// set `ComputeObjective` according to `penalty_type`:
// `'d'` for default, i.e., objective given by
// `.5||R - RX||_F^2 + lambda * ||RX||_*`;
// `'a'` for alternative, i.e., objective given by
// `.5||R - RX||_F^2 + lambda * ||X||_*`. Default is `'d'`.
std::function<double(void)> MarkovitzRRRSolver::SetObjectiveFunction(
  const char penalty_type,
  const double lambda1,
  const double lambda2
) const {

  switch (penalty_type) {

  // default Nuclear penalty: `lambda1 * ||R * X||_*`
  case 'd': {

    // if both `lambda1` and `lambda2` are greater than `0`,
    // use both Nuclear and Ridge penalty
    if ((lambda1 > 0.) & (lambda2 > 0.)) {


      return (double)N/T >= .7 ?
      std::bind(&MarkovitzRRRSolver::ComputeMainObjectiveNuclearRidge, this) :
      std::bind(&MarkovitzRRRSolver::ComputeMainObjectiveNuclearRidgeSmallN, this);

    }

    // if only `lambda1 > 0`, use the Nuclear penalty
    if (lambda1 > 0.) {

      return (double)N/T >= .7 ?
      std::bind(&MarkovitzRRRSolver::ComputeMainObjectiveNuclear, this) :
      std::bind(&MarkovitzRRRSolver::ComputeMainObjectiveNuclearSmallN, this);

    }

    // otherwise if only `lambda2 > 0`, use the Ridge penalty
    return std::bind(
      &MarkovitzRRRSolver::ComputeMainObjectiveRidge,
      this
    );

  }

  // alternative Nuclear penalty: `lambda1 * ||X||_*`
  default: {

    // if both `lambda1` and `lambda2` are greater than `0`,
    // use both the alternative Nuclear and the Ridge penalty
    if ((lambda1 > 0.) & (lambda2 > 0.)) {

      return std::bind(
        &MarkovitzRRRSolver::ComputeMainObjectiveAlternativeNuclearRidge,
        this
      );

    }

    // if only `lambda1 > 0`, use the alternative Nuclear penalty
    if (lambda1 > 0.) {

      return std::bind(
        &MarkovitzRRRSolver::ComputeMainObjectiveAlternativeNuclear,
        this
      );

    }

    // otherwise if only `lambda2 > 0`, use the Ridge penalty
    return std::bind(
      &MarkovitzRRRSolver::ComputeMainObjectiveRidge,
      this
    );

  }

  }

};

// set `ComputeSubgradient` according to `penalty_type`:
// `'d'` for "default", i.e., `lambda1 ||R * X||_* + lambda2/2 ||X||_F^2`,
// and `'a'` for "alternative", i.e., `lambda ||X||_* + lambda2/2 ||X||_F^2`.
std::function<void(void)> MarkovitzRRRSolver::SetSubgradientFunction(
  const char penalty_type,
  const double lambda1,
  const double lambda2
) {

  switch (penalty_type) {

  // default Nuclear penalty: `lambda1 * ||R * X||_*`
  case 'd': {

    // if both `lambda1` and `lambda2` are greater than `0`,
    // use both Nuclear and Ridge penalty
    if ((lambda1 > 0.) & (lambda2 > 0.)) {


      return (double)N/T >= .7 ?
      std::bind(&MarkovitzRRRSolver::ComputeSubgradientMainObjectiveNuclearRidge, this) :
      std::bind(&MarkovitzRRRSolver::ComputeSubgradientMainObjectiveNuclearRidgeSmallN, this);

    }

    // if only `lambda1 > 0`, use the Nuclear penalty
    if (lambda1 > 0.) {

      return (double)N/T >= .7 ?
      std::bind(&MarkovitzRRRSolver::ComputeSubgradientMainObjectiveNuclear, this) :
      std::bind(&MarkovitzRRRSolver::ComputeSubgradientMainObjectiveNuclearSmallN, this);

    }

    // otherwise if only `lambda2 > 0`, use the Ridge penalty
    return std::bind(
      &MarkovitzRRRSolver::ComputeSubgradientMainObjectiveRidge,
      this
    );

  }

  // alternative Nuclear penalty: `lambda1 * ||X||_*`
  default: {

    // if both `lambda1` and `lambda2` are greater than `0`,
    // use both the alternative Nuclear and the Ridge penalty
    if ((lambda1 > 0.) & (lambda2 > 0.)) {

      return std::bind(
        &MarkovitzRRRSolver::ComputeSubgradientMainObjectiveAlternativeNuclearRidge,
        this
      );

    }

    // if only `lambda1 > 0`, use the alternative Nuclear penalty
    if (lambda1 > 0.) {

      return std::bind(
        &MarkovitzRRRSolver::ComputeSubgradientMainObjectiveAlternativeNuclear,
        this
      );

    }

    // otherwise if only `lambda2 > 0`, use the Ridge penalty
    return std::bind(
      &MarkovitzRRRSolver::ComputeSubgradientMainObjectiveRidge,
      this
    );

  }

  }

};

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

// set the step size constant to `step_size_constant` if `step_size_constant > 0`
// otherwise set it to `2./(min(sv(R))^2 + max(sv(R))^2)`, where `sv` denotes
// singular values
double MarkovitzRRRSolver::SetStepSizeConstant(
  const double step_size_constant,
  const double lambda2
) const {

  if (step_size_constant > 0.) return step_size_constant;

  const arma::vec svR = arma::svd(R);
  return 2. / (svR(0) * svR(0) + svR(minNT - 1) * svR(minNT - 1) + lambda2);

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
const arma::rowvec& MarkovitzRRRSolver::GetObjective() const {
  return objective;
};

// get solution
const arma::mat& MarkovitzRRRSolver::GetSolution() const {
  return Xbest;
};

// get the optimal portfolio weights
const arma::rowvec& MarkovitzRRRSolver::GetWeights() const {
  return weights;
};

// get number of iterations
const unsigned int MarkovitzRRRSolver::GetIterations() const {
  return iter;
};

// ancilliary

// get
const arma::vec& MarkovitzRRRSolver::GetX_norm() const {
  return X_norm;
};

// get
const arma::vec& MarkovitzRRRSolver::GetSigma_inv_norm() const {
  return Sigma_inv_norm;
};

// get
const arma::mat& MarkovitzRRRSolver::GetWWeights() const {
  return Weights;
};
