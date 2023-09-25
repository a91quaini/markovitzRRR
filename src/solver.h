// Author: Alberto Quaini

#ifndef MARKOVITZRRRSOLVER_H
#define MARKOVITZRRRSOLVER_H

#include <RcppArmadillo.h>

//// Definition of MarkovitzRRRSolver

class MarkovitzRRRSolver {
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
  std::vector<double> objective;
  // subgradient
  arma::mat subgradient;
  // tolerance
  const double tolerance;
  // solver status
  std::string status;
  // solver parameters
  const double step_size_constant;
  // function computing the step size
  std::function<double(int)> ComputeStepSize;

  // accessible members
public:

  // class constructor
  explicit MarkovitzRRRSolver(
    const arma::mat& R,
    const arma::mat& X0,
    const double lambda,
    const char step_size_type,
    const double step_size_constant,
    const double tolerance
  );

  // Compute the projected subgradient step based on the current iteration
  void ProjectedSubgradientStep(unsigned int iter);

  // compute the subgradient at X0
  void ComputeSubgradient();

  // compute the objective function at a given X
  double ComputeObjective(const arma::mat& X) const;

  // set the penalty parameter lambda
  void SetLambda(double lambda);

  // set the X0 matrix to an hollow matrix with 1/N in the off-diagonal
  void SetX0ToHollow1OverN();

  // get the vector of objective function evaluation at each iteration
  const std::vector<double>& GetObjective() const;

  // get solver status "unsolved" or "solved"
  std::string GetStatus() const;

  // get the solver solution
  arma::mat GetSolution() const;

  // compute the optimal portfolio weights
  arma::rowvec GetOptimalPortfolioWeights() const;

};

#endif
