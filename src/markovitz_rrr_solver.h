// Author: Alberto Quaini

#ifndef MARKOVITZRRRSOLVER_H
#define MARKOVITZRRRSOLVER_H

#include <RcppArmadillo.h>

//// Definition of MarkovitzRRRSolver

class MarkovitzRRRSolver {
  // optimization solver for following optimization problem over R^(NxN)
  // minimize_X {0.5 ||R - RX||_F^2 + lambda1 ||RX||_* + lambda2/2 ||X||_F^2 | diag(X) = 0}
  // or the alternative
  // minimize_X {0.5 ||R - RX||_F^2 + lambda1 ||X||_* + lambda2/2 ||X||_F^2 | diag(X) = 0},
  // where ||.||_F denotes the Frobenious norm and ||.||_* the nuclear norm

  // members directly accessible only inside the class
private:

  //// data
  // returns
  const arma::mat& returns;
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
  const double lambda1;
  const double lambda2;
  // objective function
  const std::function<double(void)> ComputeObjective;
  arma::rowvec objective;
  double objective_best;
  // subgradient
  const std::function<void(void)> ComputeSubgradient;
  arma::mat subgradient;
  // useful for subgradient
  arma::mat U, V;
  arma::vec sv;
  // step size
  const double step_size_constant;
  const std::function<double(void)> ComputeStepSize;
  // iterations
  const unsigned int max_iter;
  unsigned int iter;
  // tolerance
  const double tolerance;
  // solver status
  bool is_improved;
  bool is_converged;

  // accessible members
public:

  // class constructor

  explicit MarkovitzRRRSolver(
    const arma::mat& returns,
    const arma::mat& X0,
    const double lambda1 = 0.,
    const double lambda2 = 0.,
    const char penalty_type = 'd',
    const char step_size_type = 'd',
    const double step_size_constant = 0.,
    const unsigned int max_iter = 10000,
    const double tolerance = 0.
  );

  // Solve the unpenalized Markovitz optimization problem
  void SolveUnpenalizedMarkovitz();

  // Solve the optimization problem using the projected subgradient path
  void Solve();

  // Compute the projected subgradient step based on the current iteration
  void ComputeProjectedSubgradientStep();

  // Compute the initial objective function
  void ComputeInitialObjective();

  // set the X0 matrix to an hollow matrix with 1/N in the off-diagonal
  void SetX0ToHollow1OverN();

  // compute the optimal portfolio weights
  void ComputeOptimalPortfolioWeights();

  // Check solver status:
  // is the objective value decreased from the value at the initial value?
  // is the objective value at the last solution equal to
  // the value at the best solution?
  void CheckSolverStatus();

  // compute the objective function at a given `X`
  // depending on the values of `penalty_type`, `lambda1` and `lambda2`
  double ComputeMainObjectiveRidge() const;
  double ComputeMainObjectiveNuclear() const;
  double ComputeMainObjectiveNuclearSmallN() const;
  double ComputeMainObjectiveNuclearRidge() const;
  double ComputeMainObjectiveNuclearRidgeSmallN() const;
  double ComputeMainObjectiveAlternativeNuclear() const;
  double ComputeMainObjectiveAlternativeNuclearRidge() const;

  // compute `step_size` at the current iteration
  double ComputeStepSizeConstant() const;
  double ComputeStepSizeNotSummableVanishing() const;
  double ComputeStepSizeSquareSummableNotSummable() const;
  double ComputeStepSizeConstantStepLength() const;
  double ComputeStepSizeModifiedPolyak() const;

  // compute `subgradient` at a given `X0`
  // depending on the values of `penalty_type`, `lambda1` and `lambda2`
  void ComputeSubgradientMainObjectiveRidge();
  void ComputeSubgradientMainObjectiveNuclear();
  void ComputeSubgradientMainObjectiveNuclearSmallN();
  void ComputeSubgradientMainObjectiveNuclearRidge();
  void ComputeSubgradientMainObjectiveNuclearRidgeSmallN();
  void ComputeSubgradientMainObjectiveAlternativeNuclear();
  void ComputeSubgradientMainObjectiveAlternativeNuclearRidge();

  //// setters

  // set function computing the objective function at the current iteration
  std::function<double(void)> SetObjectiveFunction(
    const char penalty_type,
    const double lambda1,
    const double lambda2
  ) const;

  // set function computing the `step_size` at the current iteration
  std::function<double(void)> SetStepSizeFunction(
    const char step_size_type
  ) const;

  // set `step_size_constant`
  double SetStepSizeConstant(
    const double step_size_constant,
    const double lambda2
  ) const;

  // set function computing the subgradient of the objective function at `X0`
  std::function<void(void)> SetSubgradientFunction(
    const char penalty_type,
    const double lambda1,
    const double lambda2
  );

  //// getters

  // get the vector of objective function evaluation at each iteration
  const arma::rowvec& GetObjective() const;

  // get the solver solution
  const arma::mat& GetSolution() const;

  // get the optimal portfolio weights
  const arma::rowvec& GetWeights() const;

  // get number of iterations
  const unsigned int GetIterations() const;

  // get solver status:
  // is the objective value decreased from the value at the initial value?
  // is the objective value at the last solution equal to
  // the value at the best solution?
  const bool GetIsImproved() const;
  const bool GetIsConverged() const;

  // get output list
  const Rcpp::List GetOutputList() const;

};

#endif
