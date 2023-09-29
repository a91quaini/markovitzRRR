// Author: Alberto Quaini

#ifndef MARKOVITZRRRSOLVER_H
#define MARKOVITZRRRSOLVER_H

#include <RcppArmadillo.h>

//// Definition of MarkovitzRRRSolver

class MarkovitzRRRSolver {
  // optimization solver for following optimization problem over R^(NxN)
  // minimize_X {0.5 ||R - R * X||_F^2 + lambda ||R * X||_* | diag(X) = 0}
  // or the alternative
  // minimize_X {0.5 ||R - R * X||_F^2 + lambda ||X||_* | diag(X) = 0},
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
  // penalty parameter
  double lambda;
  // objective function
  arma::vec objective;
  // subgradient
  arma::mat subgradient;
  // function computing the subgradient
  std::function<void(void)> ComputeSubgradient;
  // useful for subgradient
  arma::mat U, V;
  arma::vec sv;
  // step size
  const double step_size_constant;
  // function computing the step size
  std::function<double(void)> ComputeStepSize;
  // iterations
  const unsigned int max_iter;
  unsigned int iter;
  // // solver status
  // std::string status;
  // // tolerance
  // const double tolerance;

  // accessible members
public:

  // class constructor

  explicit MarkovitzRRRSolver(
    const arma::mat& R,
    const arma::mat& X0,
    const double lambda,
    const char penalty_type = 'd',
    const char step_size_type = 'd',
    const double step_size_constant = 1.e-3,
    const unsigned int max_iter = 10000
    // const double tolerance
  );

  // Solve the optimization problem using the projected subgradient path
  void Solve();

  // Compute the projected subgradient step based on the current iteration
  void ComputeProjectedSubgradientStep(const unsigned int iter);

  // compute the objective function at a given X
  double ComputeObjective(const arma::mat& X) const;

  // compute `step_size` at the current iteration
  std::function<double(void)> SetStepSizeFunction(const char step_size_type) const;
  double ComputeStepSizeConstant() const;
  double ComputeStepSizeNotSummableVanishing() const;
  double ComputeStepSizeSquareSummableNotSummable() const;
  double ComputeStepSizeModifiedPolyak() const;

  // compute `subgradient` at the current iteration
  std::function<void(void)> SetSubgradientFunction(const char penalty_type);
  void ComputeSubgradientForLargeN();
  void ComputeSubgradientForSmallN();
  void ComputeSubgradientAlternative();

  // set the penalty parameter lambda
  void SetLambda(double lambda);

  // set the X0 matrix to an hollow matrix with 1/N in the off-diagonal
  void SetX0ToHollow1OverN();

  // get the vector of objective function evaluation at each iteration
  const arma::vec& GetObjective() const;

  // get the solver solution
  const arma::mat& GetSolution() const;

  // compute the optimal portfolio weights
  arma::rowvec GetOptimalPortfolioWeights() const;

};

#endif
