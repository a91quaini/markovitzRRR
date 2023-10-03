// Author: Alberto Quaini

#ifndef MARKOVITZRRRSOLVER_H
#define MARKOVITZRRRSOLVER_H

#include <RcppArmadillo.h>

//// Definition of constants used by the solver
const char default_choice_type = 'd';
const double default_step_size_constant = 1.e-3;
const unsigned int default_max_iter = 10000;
const double default_tolerance = -1.;

//// Definition of MarkovitzRRRSolver

class MarkovitzRRRSolver {
  // optimization solver for following optimization problem over R^(NxN)
  // minimize_X {0.5 ||R - RX||_F^2 + lambda ||RX||_* | diag(X) = 0}
  // or the alternative
  // minimize_X {0.5 ||R - RX||_F^2 + lambda ||X||_* | diag(X) = 0},
  // where ||.||_F denotes the Frobenious norm and ||.||_* the nuclear norm

  // members directly accessible only inside the class
private:

  //// data
  // returns
  const arma::mat R;
  // returns dimensions
  const unsigned int T;
  const unsigned int N;

  //// solver internal data
  // solutions
  arma::mat X0;
  arma::mat X1;
  arma::mat Xbest;
  // optimal portfolio weights
  arma::rowvec weights;
  // penalty parameter
  double lambda;
  // objective function
  arma::vec objective;
  const std::function<double(const arma::mat&)> ComputeObjective;
  // subgradient
  arma::mat subgradient;
  // function computing the subgradient
  const std::function<void(void)> ComputeSubgradient;
  // useful for subgradient
  arma::mat U, V;
  arma::vec sv;
  // step size
  const double step_size_constant;
  // function computing the step size
  const std::function<double(void)> ComputeStepSize;
  // iterations
  const unsigned int max_iter;
  unsigned int iter;
  // tolerance
  const double tolerance;

  // accessible members
public:

  // class constructor

  explicit MarkovitzRRRSolver(
    const arma::mat& R,
    const arma::mat& X0,
    const double lambda,
    const char objective_type = default_choice_type,
    const char penalty_type = default_choice_type,
    const char step_size_type = default_choice_type,
    const double step_size_constant = default_step_size_constant,
    const unsigned int max_iter = default_max_iter,
    const double tolerance = default_tolerance
  );

  // Solve the optimization problem using the projected subgradient path
  void Solve();

  // Compute the projected subgradient step based on the current iteration
  void ComputeProjectedSubgradientStep(const unsigned int iter);

  // compute the objective function at a given X
  std::function<double(const arma::mat&)> SetObjectiveFunction(
    const char objective_type
  ) const;
  double ComputeDefaultObjective(const arma::mat& X) const;
  double ComputeAlternativeObjective(const arma::mat& X) const;

  // compute `step_size` at the current iteration
  std::function<double(void)> SetStepSizeFunction(
    const char step_size_type
  ) const;
  double ComputeStepSizeConstant() const;
  double ComputeStepSizeConstantStepLength() const;
  double ComputeStepSizeNotSummableVanishing() const;
  double ComputeStepSizeSquareSummableNotSummable() const;
  double ComputeStepSizeModifiedPolyak() const;

  // compute `subgradient` at the current iteration
  std::function<void(void)> SetSubgradientFunction(const char penalty_type);
  void ComputeSubgradientForLargeN();
  void ComputeSubgradientForSmallN();
  void ComputeSubgradientAlternative();

  // compute the optimal portfolio weights
  arma::rowvec ComputeOptimalPortfolioWeights();

  // set the penalty parameter lambda
  void SetLambda(double lambda);

  // set the X0 matrix to an hollow matrix with 1/N in the off-diagonal
  void SetX0ToHollow1OverN();

  // get the vector of objective function evaluation at each iteration
  const arma::vec& GetObjective() const;

  // get the solver solution
  const arma::mat& GetSolution() const;

};

#endif
