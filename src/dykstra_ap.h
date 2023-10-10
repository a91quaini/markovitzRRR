// Author: Alberto Quaini

#ifndef DYKSTRAAP_H
#define DYKSTRAAP_H

#include <RcppArmadillo.h>
#include "constants.h"

//// Dykstra's Alternating Projection solver

class DykstraAP {
  // optimization solver for finding solving:
  // min_X {||R - RX||_F | X in A intersection B}
  // where, for given tau > 0:
  // A = {RX | ||RX||_* <= tau}
  // B = {RX | diag(X)=0} or
  // B = {RX | diag(X)=0, ||Xi||_F <= lambda, for i = 1,...,N}
  // where ||.||_F denotes the Frobenious norm and ||.||_* the nuclear norm.

  // members directly accessible only inside the class
private:

  //// data
  // returns
  const arma::mat R;
  // returns dimensions
  const unsigned int T;
  const unsigned int N;

  //// solver internal data
  // solution
  arma::mat X;
  // projection on A and B
  arma::mat PA;
  arma::mat PB;
  // increments on A and B
  arma::mat IA;
  arma::mat IB;
  // // store Ri'Ri, where Ri is matrix R without the i-th column
  // const std::vector<arma::mat> RiRi;
  // columns index vector
  arma::uvec assets_idx;
  // optimal portfolio weights
  arma::rowvec weights;
  // problem parameters
  double lambda;
  double tau;
  // objective function
  std::vector<double> objective;
  // svd decomposition
  arma::mat U, V;
  arma::vec sv;
  // iterations
  const unsigned int max_iter;
  // tolerance
  const double tolerance;

  // accessible members
public:

  // class constructor

  explicit DykstraAP(
    const arma::mat& R,
    const double tau,
    const double lambda = minus_one,
    const unsigned int max_iter = default_max_iter,
    const double tolerance = minus_one
  );

  // Solve the optimization problem
  void Solve();

  // Compute the projection of `PB + IA` on A = {RX | ||RX||_* <= tau}
  // and update `IA`.
  void StepA();

  // Compute the projection of `PA + IB` on `B = {RX | diag(X)=0}`
  // and update `IB`.
  void StepB();

  // compute the objective function
  double ComputeObjective() const;

  // // compute the optimal portfolio weights
  // arma::rowvec ComputeOptimalPortfolioWeights();

  // // set RiRi, where Ri is matrix R without the i-th column
  // const std::vector<arma::mat>& SetRiRi();

  // set the penalty parameter lambda
  void SetLambda(double lambda);

  // set the X0 matrix to an hollow matrix with 1/N in the off-diagonal
  void SetX0ToHollow1OverN();

  // get the vector of objective function evaluation at each iteration
  const std::vector<double>& GetObjective() const;

  // get the solver solution
  const arma::mat& GetSolution() const;

};

#endif