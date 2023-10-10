// Author: Alberto Quaini

#include "dykstra_ap.h"

//// Implementation of MarkovitzRRRSolver

// class constructor
DykstraAP::DykstraAP(
  const arma::mat& R,
  const double tau,
  const double lambda,
  const unsigned int max_iter,
  const double tolerance
) : // initialize object members at class construction
  R(R),
  T(R.n_rows),
  N(R.n_cols),
  X(arma::zeros(N, N)),
  PB(R),
  IA(arma::zeros(T, N)),
  IB(arma::zeros(T, N)),
  // RiRi(SetRiRi()),
  assets_idx(arma::regspace<arma::uvec>(0, N-1)),
  lambda(lambda),
  tau(tau),
  max_iter(max_iter),
  tolerance(tolerance)
{

  // compute the objective function at `X`
  objective.push_back(ComputeObjective());

  // // if N is small compared to T, the `ComputeSubgradient` function assumes
  // // that the svd of R is computed and stored in U, sv and V
  // if ((double)N/T < .7) {
  //
  //   // compute svd(R) once
  //   arma::svd(U, sv, V, R);
  //   // remove the last (T - N) columns from U
  //   U.shed_cols(N, T-1);
  //
  // }

};

// Solve the optimization problem
void DykstraAP::Solve() {

  unsigned int iter = 0;

  // main loop with solution check
  if (tolerance > 0) {

    while(++iter < max_iter) {

      // store current increments IA and IB
      const arma::mat IA0 = IA;
      const arma::mat IB0 = IB;

      // Compute the projection of `PB + IA` on A = {RX | ||RX||_* <= tau}
      // and update `IA`.
      StepA();

      // Compute the projection of `PA + IB` on `B = {RX | diag(X)=0}`
      // and update `IB`.
      StepB();

      // compute the objective function at `X`
      objective.push_back(ComputeObjective());

      // if `||IA - IA0||_F^2 + ||IB - IB0||_F^2 < tolerance`, quit loop.
      if (arma::accu(arma::square(IA - IA0)) +
            arma::accu(arma::square(IB - IB0)) < tolerance) {
        break;
      }

    }

  // main loop without solution check
  } else {

    // main loop
    while(++iter < max_iter) {

      // Compute the projection of `PB + IA` on A = {RX | ||RX||_* <= tau}
      // and update `IA`.
      StepA();

      // Compute the projection of `PA + IB` on `B = {RX | diag(X)=0}`
      // and update `IB`.
      StepB();

      // compute the objective function at `X`
      objective.push_back(ComputeObjective());

    }

  }

}

// Compute the projection of `PB + IA` on A = {RX | ||RX||_* <= tau}
// and update `IA`.
void DykstraAP::StepA() {

  // svd decomposition of `PB + IA`
  // note: in the first iteration, `PB = R` and `IA = 0`
  svd(U, sv, V, PB + IA);

  // project the singular values of `PB + IA` onto the unit simplex
  sv *= tau / arma::accu(sv);
  sv.clamp(0., arma::datum::inf);

  // compute PA with soft-thresholded singular values
  PA = U.head_cols(N) * arma::diagmat(sv) * V.t();

  // increment IA
  // IA += PB - PA;

};

// Compute the projection of `PA + IB` on `B = {RX | diag(X)=0}` and update `IB`.
// to do so: solve N regressions of `Y(i)`, with `Y = PA + IB`, on `R^(-i)`,
// where `Y(i)` is the i-th column of `Y` and `R^(-i)` is `R` without the i-th column,
// then store appropriately the computed regression coefficients
// along the non-diagonal elements of `X`.
void DykstraAP::StepB() {

  for (unsigned int i=0; i<N; ++i) {

    // const arma::uvec columns = join_vert(
    //   assets_idx.head(i),
    //   assets_idx.tail(N-1-i)
    // );

    arma::mat Ri = R;
    Ri.shed_col(i);
    const arma::vec coeff = arma::solve(
      Ri.t() * Ri, //+ lambda * arma::eye(N-1, N-1),
      Ri.t() * (PA.col(i) + IB.col(i)),
      arma::solve_opts::likely_sympd
    );

    // insert the solution in elements 1,...,i-1,i+1,..,N of the
    // i-th column of X
    X.col(i) = arma::join_vert(
      coeff.head(i),
      arma::zeros(1),
      coeff.tail(N-1-i)
    );
    // X(arma::uvec(1, arma::fill::value(i), columns) = coeff;

  }

  // compute PB
  PB = R * X;

  // update `IB`
  // IB += PA - PB;

};

// compute objective value for given X
double DykstraAP::ComputeObjective() const {

  // return the objective function at X --> ||R - RX||_F^2
  return arma::accu(arma::square(R - R * X));

};


///////////////
/// setters ///

// const std::vector<arma::mat>& DykstraAP::SetRiRi() {
//
//
//   for (unsigned int i=0; i<N; ++i) {
//
//     arma::mat Ri = R;
//     Ri.shed_col(i);
//     RiRi.push_back(Ri.t() * Ri);
//
//   }
//
//   return RiRi;
//
// }

// set lambda
void DykstraAP::SetLambda(const double lambda) {
  this->lambda = lambda;
};

///////////////
/// getters ///

// get objective
const std::vector<double>& DykstraAP::GetObjective() const {
  return objective;
};

// get solution
const arma::mat& DykstraAP::GetSolution() const {
  return X;
};
