// Author: Alberto Quaini

#ifndef MARKOVITZRRRSOLVER_H
#define MARKOVITZRRRSOLVER_H

#include <RcppArmadillo.h>
#include "constants.h"

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
  const unsigned int minNT;

  //// solver internal data
  // solutions
  arma::mat X;
  arma::mat X0;
  arma::mat Xbest;
  // optimal portfolio weights
  arma::rowvec weights;
  // penalty parameter
  double lambda;
  // iterations
  const unsigned int max_iter;
  unsigned int iter;
  // objective function
  const char penalty_type;
  const std::function<double()> ComputeObjective;
  arma::rowvec objective;
  double objective_best;
  // subgradient
  const std::function<void(void)> ComputeSubgradient;
  arma::mat subgradient;
  // useful for subgradient
  arma::mat U, V;
  arma::vec sv;
  // step size
  const char step_size_type;
  const double step_size_constant;
  const std::function<double(void)> ComputeStepSize;
  // tolerance
  const double tolerance;
  // ancilliary
  arma::vec X_norm;
  arma::vec Sigma_inv_norm;
  arma::mat Weights;

  // accessible members
public:

  // class constructor

  explicit MarkovitzRRRSolver(
    const arma::mat& R,
    const arma::mat& X0,
    const double lambda,
    const char penalty_type = default_choice_type,
    const char step_size_type = default_choice_type,
    const double step_size_constant = minus_one,
    const unsigned int max_iter = default_max_iter,
    const double tolerance = minus_one
  );

  // Solve the optimization problem using the projected subgradient path
  void Solve();

  // Compute the projected subgradient step based on the current iteration
  void ComputeProjectedSubgradientStep();

  // compute the objective function at a given X
  double ComputeDefaultObjective() const;
  double ComputeAlternativeObjective() const;

  // compute `step_size` at the current iteration
  double ComputeStepSizeConstant() const;
  double ComputeStepSizeConstantStepLength() const;
  double ComputeStepSizeNotSummableVanishing() const;
  double ComputeStepSizeSquareSummableNotSummable() const;
  double ComputeStepSizeModifiedPolyak() const;

  // compute `subgradient` at the current iteration
  void ComputeSubgradientForLargeN();
  void ComputeSubgradientForSmallN();
  void ComputeSubgradientAlternative();

  // compute the optimal portfolio weights
  void ComputeOptimalPortfolioWeights();

  //// setters

  // set function computing the `step_size` at the current iteration
  std::function<double(void)> SetObjectiveFunction() const;
  std::function<double(void)> SetStepSizeFunction() const;
  std::function<void(void)> SetSubgradientFunction();

  // set `step_size_constant`
  double SetStepSizeConstant(const double step_size_constant) const;

  // set the penalty parameter lambda
  void SetLambda(double lambda);

  // set the X0 matrix to an hollow matrix with 1/N in the off-diagonal
  void SetX0ToHollow1OverN();

  //// getters

  // get the vector of objective function evaluation at each iteration
  const arma::rowvec& GetObjective() const;

  // get the solver solution
  const arma::mat& GetSolution() const;

  // get the optimal portfolio weights
  const arma::rowvec& GetWeights() const;

  // get number of iterations
  const unsigned int GetIterations() const;

  // get
  const arma::vec& GetX_norm() const;

  // get
  const arma::vec& GetSigma_inv_norm() const;

  // get
  const arma::mat& GetWWeights() const;

};

#endif
