// Author: Alberto Quaini

#include "markovitz_rrr_solver.h"

// Class constructor
MarkovitzRRRSolver::MarkovitzRRRSolver(
  const arma::mat& returns,
  const arma::mat& X0,
  const double lambda1,
  const double lambda2,
  const char penalty_type,
  const char step_size_type,
  const double step_size_constant,
  const unsigned int max_iter,
  const double tolerance
) :
returns(returns),
T(returns.n_rows),
N(returns.n_cols),
minNT(std::min(N, T)),
X(X0),
Xbest(X0),
lambda1(lambda1),
lambda2(lambda2),
ComputeObjective(SetObjectiveFunction(penalty_type, lambda1, lambda2)),
objective(max_iter),
ComputeSubgradient(SetSubgradientFunction(penalty_type, lambda1, lambda2)),
step_size_constant(SetStepSizeConstant(step_size_constant, lambda2)),
ComputeStepSize(SetStepSizeFunction(step_size_type)),
max_iter(max_iter),
iter(0),
tolerance(tolerance),
is_improved(false),
is_converged(false) {

  ComputeInitialObjective();

}

// compute the projected subgradient optimization path
void MarkovitzRRRSolver::Solve() {

  // Solve the unpenalized problem if lambda1 and lambda2 are zero or negative
  // and return the optimal portfolio weights
  if ((lambda1 <= 0.) && (lambda2 <= 0.)) {

    SolveUnpenalizedMarkovitz();
    ComputeOptimalPortfolioWeights();
    return;

  }

  // Otherwise solve the penalized problem via the projected subgradient method
  while(++iter < max_iter) {

    // set `X0` to `X`
    X0 = X;

    // Compute the projected subgradient step based on the current iteration
    ComputeProjectedSubgradientStep();

    // If tolerance > 0, then compute a stop rule based on the Frobenius
    // distance between consecutive solutions
    if (
        (tolerance > 0) &&
          (arma::norm(X - X0, "fro") / static_cast<double>(N) <= tolerance)
    ) break;

    // Until the number of iterations hits the maximum number
  };

  // Compute optimal portfolio weights
  ComputeOptimalPortfolioWeights();

  // Check and update solver status
  CheckSolverStatus();

}

// solve the unpenalized Markovitz optimization problem
void MarkovitzRRRSolver::SolveUnpenalizedMarkovitz() {


  // compute `X` by running N linear regressions of the i-th column of R
  // on the remaining (hedging) columns of R
  for (unsigned int i = 0; i < N; ++i) {

    // Set indices of hedging portfolios
    arma::uvec indices;
    if (i < N - 1) {
      // Standard case: concatenate two ranges
      indices = arma::join_vert(
        arma::regspace<arma::uvec>(0, i - 1),
        arma::regspace<arma::uvec>(i + 1, N - 1)
      );
    } else {
      // Edge case: i is the last index, so only take the first range
      indices = arma::regspace<arma::uvec>(0, N - 2);  // N - 2 is the last valid index before N - 1
    }

    // Store hedging portfolio returns
    const arma::mat returns_i = returns.cols(indices);

    // Compute hedging coefficients
    const arma::vec coeff = arma::solve(
      returns_i.t() * returns_i,
      returns_i.t() * returns.col(i),
      arma::solve_opts::likely_sympd
    );

    // Fill in the solution coefficient matrix
    Xbest.col(i) = arma::join_vert(
      coeff.head(i),
      arma::zeros(1),
      coeff.tail(N-1-i)
    );

  }

  // Store the objective function value `1/2||R - R * X||_F^2`
  objective(1) = .5 * arma::accu(arma::square(returns - returns * X));
  objective_best = std::min(objective(0), objective(1));

  // Keep only the first two function evaluations
  objective.resize(2);

  // Check and update solver status
  CheckSolverStatus();

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

void MarkovitzRRRSolver::ComputeInitialObjective() {

  // if N is small compared to T, the `ComputeSubgradient` function assumes
  // that the svd of R is computed and stored in U, sv and V
  // note: here `default_NT_ratio = .4`
  if (static_cast<double>(N) / T < .4) {

    // compute svd(R) once
    arma::svd(U, sv, V, returns);
    // remove the last (T - N) columns from U
    U.shed_cols(N, T - 1);

  }

  // compute the objective function at `X0`
  objective(0) = ComputeObjective();

  // set the best objective value to `objective(0)`
  objective_best = objective(0);

}

// set initial point to hollow matrix with 1/N on the off-diagonal
void MarkovitzRRRSolver::SetX0ToHollow1OverN () {

  X0 = arma::toeplitz(arma::join_cols(
    arma::vec::fixed<1>(arma::fill::zeros),
    arma::vec(returns.n_cols - 1, arma::fill::value(1. / returns.n_cols))
  ));

};

// Check solver status
void MarkovitzRRRSolver::CheckSolverStatus() {

  // Do not update solver_status if the objective is empty or if the objective
  // at the best solution is not lower than the value at the initial point
  if (objective.empty()) return;

  // Set is_improved to true if the objective value is decreased from the
  // value at the initial point
  is_improved = objective_best <= objective(0);

  // Set is_converged to true if the objective value at the last solution equals
  // the value at the best solution
  is_converged = objective(objective.n_elem - 1) <= arma::min(objective) +
    .5 * arma::stddev(objective);
    // 1e2 * arma::datum::eps;

}

// compute the optimal portfolio weights
void MarkovitzRRRSolver::ComputeOptimalPortfolioWeights() {

  // Compute the inverse of the variances of residuals
  const arma::rowvec inv_var_res = 1.0 / arma::max(
    arma::var(returns - returns * Xbest),
    arma::datum::eps * 2.0
  );

  // compute the unscaled optimal weights: Sigma^(-1)'1 where Sigma^(-1),
  // the inverse variance covariance matrix of returns, is computed via
  // the solution X and the marginal variances of the residuals
  // weights = arma::sum(
  //   arma::diagmat(inv_var_res) * (arma::eye(N, N) - Xbest), 1
  // );
  weights = arma::trans(arma::sum(arma::eye(N, N) - Xbest, 0) % inv_var_res);

  // Compute the sum of the weights
  const double weights_sum = arma::accu(weights);

  // Project weights into the unit simplex
  if (weights_sum != 0) weights /= weights_sum;

};


//// Objective function

// compute the objective value with Ridge penalty:
// `1/2 ||R - RX||_F^2 + lambda2/2||X||_F^2` for given `X`
double MarkovitzRRRSolver::ComputeMainObjectiveRidge() const {

  // Rcpp::Rcout << "ComputeMainObjectiveRidge\n";
  return .5 * arma::accu(arma::square(returns - returns * X)) +
         .5 * lambda2 * arma::accu(X % X);

};

// compute the objective function with Nuclear penalty:
// `1/2 ||R - RX||_F^2 + lambda1 ||RX||_*` for given `X`
double MarkovitzRRRSolver::ComputeMainObjectiveNuclear() const {

  // Rcpp::Rcout << "ComputeMainObjectiveNuclear\n";
  const arma::mat returnsX = returns * X;
  return .5 * arma::accu(arma::square(returns - returnsX)) +
    lambda1 * arma::accu(arma::svd(returnsX));

};

// compute the objective function with Nuclear penalty when N is "small":
// `1/2 ||R - RX||_F^2 + lambda1 ||RX||_*` for given `X`
double MarkovitzRRRSolver::ComputeMainObjectiveNuclearSmallN() const {

  // compute `qr(X'V)`, where `V` contains the right singular vectors of `R`
  arma::mat Q, A;
  arma::qr(Q, A, X.t() * V);

  // return the svd decomposition of `AS`, where `S` is the diagonal matrix
  // of singular values of `R`
  return .5 * arma::accu(arma::square(returns - returns * X)) +
    lambda1 * arma::accu(arma::svd(A * arma::diagmat(sv)));

};

// compute the objective function with Nuclear and Ridfe penalties:
// `1/2 ||R - RX||_F^2 + lambda1 ||RX||_* + lambda2/2||X||_F^2` for given `X`
double MarkovitzRRRSolver::ComputeMainObjectiveNuclearRidge() const {

  // Rcpp::Rcout << "ComputeMainObjectiveNuclearRidge\n";
  const arma::mat returnsX = returns * X;
  return .5 * arma::accu(arma::square(returns - returnsX)) +
    lambda1 * arma::accu(arma::svd(returnsX)) +
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
  return .5 * arma::accu(arma::square(returns - returns * X)) +
    lambda1 * arma::accu(arma::svd(A * arma::diagmat(sv))) +
    lambda2 * arma::accu(X % X);

};

// compute the objective function with the Nuclear penalty on `X`:
// `1/2 ||R - RX||_F^2 + lambda1 ||X||_*` for given `X`
double MarkovitzRRRSolver::ComputeMainObjectiveAlternativeNuclear() const {

  // Rcpp::Rcout << "ComputeMainObjectiveAlternativeNuclear\n";
  return .5 * arma::accu(arma::square(returns - returns * X)) +
    lambda1 * arma::accu(arma::svd(X));

};

// compute the objective function with the Nuclear and Ridge penalties on `X`:
// `1/2 ||R - RX||_F^2 + lambda1 ||X||_* + lambda2/2||X||_F^2` for given `X`
double MarkovitzRRRSolver::ComputeMainObjectiveAlternativeNuclearRidge() const {

  // Rcpp::Rcout << "ComputeMainObjectiveAlternativeNuclearRidge\n";
  return .5 * arma::accu(arma::square(returns - returns * X)) +
    lambda1 * arma::accu(arma::svd(X)) +
    lambda2 * arma::accu(X % X);

};

////////////////////////
//// Subgradient ///////

// compute the subgradient of the objective value with Ridge penalty:
// `lambda2 * X0 - R'R + R'R * X0`
void MarkovitzRRRSolver::ComputeSubgradientMainObjectiveRidge() {

  // Rcpp::Rcout << "ComputeSubgradientMainObjectiveRidge\n";
  const arma::mat RtR = returns.t() * returns;
  subgradient = lambda2 * X0 - RtR + RtR * X0;

};

// compute the subgradient of the objective value with Nuclear penalty:
// `lambda1 * R'UV' - R'R + R'R * X0`, where
// `svd(R * X0) = U * S * V'`
void MarkovitzRRRSolver::ComputeSubgradientMainObjectiveNuclear() {

  // Rcpp::Rcout << "ComputeSubgradientMainObjectiveNuclear\n";
  const arma::mat RtR = returns.t() * returns;

  arma::svd(U, sv, V, returns * X0);

  subgradient = lambda1 * returns.t() * U.head_cols(minNT) *
    V.head_cols(minNT).t() + RtR * X0 - RtR;

}

// compute the subgradient of the objective value with Nuclear penalty:
// `lambda1 * R'UV' - R'R + R'R * X0`, where
// `svd(R * X0) = U * S * V'`, for "small" N
void MarkovitzRRRSolver::ComputeSubgradientMainObjectiveNuclearSmallN() {

  // Rcpp::Rcout << "ComputeSubgradientMainObjectiveNuclearSmallN\n";
  const arma::mat RtR = returns.t() * returns;

  // compute qr(X'V)
  arma::mat Q, A;
  arma::qr(Q, A, X0.t() * V);

  // compute svd(A * S)
  arma::mat U1, V1;
  arma::vec sv1;

  arma::svd(U1, sv1, V1, A * arma::diagmat(sv));

  subgradient = lambda1 * returns.t() * (U * V1) * (Q * U1).t() +
    RtR * X0 - RtR;

}

// compute the subgradient of the objective value with Nuclear and Ridge penalty:
// `lambda1 * R'UV' + lambda2 * X0 - R'R + R'R * X0`, where
// `svd(R * X0) = U * S * V'`
void MarkovitzRRRSolver::ComputeSubgradientMainObjectiveNuclearRidge() {

  // Rcpp::Rcout << "ComputeSubgradientMainObjectiveNuclearRidge\n";
  const arma::mat RtR = returns.t() * returns;

  arma::svd(U, sv, V, returns * X0);

  subgradient = lambda1 * returns.t() * U.head_cols(minNT) *
    V.head_cols(minNT).t() + lambda2 * X0 + RtR * X0 - RtR;

}

// compute the subgradient of the objective value with Nuclear and Ridge penalty:
// `lambda1 * R'UV' + lambda2 * X0 - R'R + R'R * X0`, where
// `svd(R * X0) = U * S * V'`, for N "small"
void MarkovitzRRRSolver::ComputeSubgradientMainObjectiveNuclearRidgeSmallN() {

  // Rcpp::Rcout << "ComputeSubgradientMainObjectiveNuclearRidgeSmallN\n";
  const arma::mat RtR = returns.t() * returns;

  // compute qr(X'V)
  arma::mat Q, A;
  arma::qr(Q, A, X0.t() * V);

  // compute svd(A * S)
  arma::mat U1, V1;
  arma::vec sv1;

  arma::svd(U1, sv1, V1, A * arma::diagmat(sv));

  subgradient = lambda1 * returns.t() * (U * V1) * (Q * U1).t() +
    lambda2 * X0 + RtR * X0 - RtR;

}

// compute the subgradient of the objective value with the alternative Nuclear
// penalty:
// `lambda1 * U * V' - R'R + R'R * X0`, where
// `svd(X0) = U * S * V'`
void MarkovitzRRRSolver::ComputeSubgradientMainObjectiveAlternativeNuclear() {

  // Rcpp::Rcout << "ComputeSubgradientMainObjectiveAlternativeNuclear\n";
  const arma::mat RtR = returns.t() * returns;

  arma::svd(U, sv, V, X0);

  subgradient = lambda1 * U * V.t() + RtR * X0 - RtR;

}

// compute the subgradient of the objective value with the alternative Nuclear
// and the Ridge penalty:
// `lambda1 * U * V' + lambda2 * X - R'R + R'R * X0`, where
// `svd(X0) = U * S * V'`
void MarkovitzRRRSolver::ComputeSubgradientMainObjectiveAlternativeNuclearRidge() {

  // Rcpp::Rcout << "ComputeSubgradientMainObjectiveAlternativeNuclearRidge\n";
  const arma::mat RtR = returns.t() * returns;

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

  return (
    step_size_constant + objective(iter - 1) -
    arma::min(objective.head(iter))) / arma::norm(subgradient, "fro"
  );

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


  return static_cast<double>(N) / T >= .4 ?
  std::bind(&MarkovitzRRRSolver::ComputeMainObjectiveNuclearRidge, this) :
  std::bind(&MarkovitzRRRSolver::ComputeMainObjectiveNuclearRidgeSmallN, this);

}

  // if only `lambda1 > 0`, use the Nuclear penalty
  if (lambda1 > 0.) {

    return static_cast<double>(N) / T >= .4 ?
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


  return static_cast<double>(N) / T >= .4 ?
  std::bind(&MarkovitzRRRSolver::ComputeSubgradientMainObjectiveNuclearRidge, this) :
  std::bind(&MarkovitzRRRSolver::ComputeSubgradientMainObjectiveNuclearRidgeSmallN, this);

}

  // if only `lambda1 > 0`, use the Nuclear penalty
  if (lambda1 > 0.) {

    return static_cast<double>(N) / T >= .4 ?
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

  const arma::vec sv_ret = arma::svd(returns);
  return 2. / (
    sv_ret(0) * sv_ret(0) + sv_ret(minNT - 1) * sv_ret(minNT - 1) + lambda2
  );

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
const arma::vec& MarkovitzRRRSolver::GetWeights() const {
  return weights;
};

// get number of iterations
const unsigned int MarkovitzRRRSolver::GetIterations() const {
  return iter;
};

// get solver status
const bool MarkovitzRRRSolver::GetIsImproved() const {
  return is_improved;
};
const bool MarkovitzRRRSolver::GetIsConverged() const {
  return is_converged;
};

// get output list
const Rcpp::List MarkovitzRRRSolver::GetOutputList() const {

  // Return the output list for the penalized Markovitz solution
  return Rcpp::List::create(
    Rcpp::Named("solution") = Xbest,
    Rcpp::Named("objective") = objective,
    Rcpp::Named("weights") = weights,
    Rcpp::Named("iterations") = iter,
    Rcpp::Named("is_improved") = is_improved,
    Rcpp::Named("is_converged") = is_converged
  );

};

// get output vector: useful for parallel solver
const arma::vec MarkovitzRRRSolver::GetOutputVector() const {

  const double is_converged_d = static_cast<double>(is_converged);
  const double is_improved_d = static_cast<double>(is_improved);
  // Rcpp::Rcout << "first:\n";
  // Rcpp::Rcout << arma::vec(1, arma::fill::value(lambda2)) << "\n";
  // Rcpp::Rcout << "second:\n";
  // Rcpp::Rcout << weights << "\n";
  // Rcpp::Rcout << "third:\n";
  // Rcpp::Rcout << arma::vec(1, arma::fill::value(is_improved_d)) << "\n";
  // Rcpp::Rcout << "fourth:\n";
  // Rcpp::Rcout << arma::vec(1, arma::fill::value(is_converged_d)) << "\n";

  // return weights;

  return arma::join_vert(
    arma::vec(1, arma::fill::value(lambda2)),
    weights,
    arma::vec(1, arma::fill::value(is_improved_d)),
    arma::vec(1, arma::fill::value(is_converged_d))
  );

};
