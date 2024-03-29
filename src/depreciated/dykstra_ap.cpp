// Author: Alberto Quaini

#include "dykstra_ap.h"
#include "simplexproj.h"

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
  minNT(std::min(N, T)),
  X(N, N),
  B(R),
  a(T, N),
  b(T, N),
  // RiRi(SetRiRi()),
  // assets_idx(arma::regspace<arma::uvec>(0, N-1)),
  lambda(lambda),
  tau(tau),
  max_iter(max_iter),
  iter(1),
  objective(max_iter),
  tolerance(tolerance)
{

  // compute the objective function at `X`
  objective(0) = ComputeObjective();

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

  // main loop with solution check
  if (tolerance > 0.) {

    // containers for increments a and b
    arma::mat a0;
    arma::mat b0;

    do {

      // store current increments a and b
      a0 = a;
      b0 = b;

      // Compute the projection of `B + a` on `A = {RX | ||RX||_* <= tau}`
      // and update `a`.
      ComputeStepA();

      // Compute the projection of `A + b` on `B = {RX | diag(X)=0}`
      // and update `b`.
      ComputeStepB();

      // compute the objective function at `X`
      objective(iter) = ComputeObjective();

    } while (
      // while iter < max_iter
      (++iter < max_iter) &&
      // and `||a - a0||_F^2 + ||b - b0||_F^2 < tolerance`, quit loop.
      (arma::accu(arma::square(a - a0)) +
      arma::accu(arma::square(b - b0)) > tolerance)
    );

    // remove elements in excess in `objective`
    objective = objective.head(iter);

  // main loop without solution check
  } else {

    // main loop
    do {
      Rcpp::Rcout << "\n+++++++++++++++++++++++\n";
      Rcpp::Rcout << "iter = " << iter << "\n\n";
      Rcpp::Rcout << "Step A\n\n";
      // Compute the projection of `B + a` on `A = {Z | ||Z||_* <= tau}`
      // and update `a`.
      ComputeStepA();

      Rcpp::Rcout << "Step B\n\n";
      // Compute the projection of `A + b` on `B = {RX | diag(X)=0}`
      // and update `b`.
      ComputeStepB();

      // compute the objective function at `X`
      objective(iter) = ComputeObjective();
      Rcpp::Rcout << "objective = " << objective(iter) << "\n";


    } while (++iter < max_iter);

  }

}

// Compute the projection of `B + a` on `A = {RX | ||RX||_* <= tau}`
// and update `a`.
void DykstraAP::ComputeStepA() {

  // first update `a`: `a += B`
  a += B;

  // svd decomposition of `a`
  // note: in the first iteration, `a = R`
  svd(U, sv, V, a);

  // project the singular values of `a` onto the simplex of radius `tau`
  // that is: if the sum of `sum(sv) > tau`, then multiply each `sv` by
  // `tau / sum(sv)`
  const double sv_sum = arma::accu(sv);

  Rcpp::Rcout << "sum(sv) pre = " << sv_sum << "\n";

  if (sv_sum > tau) ProjectOntoSimplex(sv, tau);
  Rcpp::Rcout << "sum(sv) post = " << arma::accu(sv) << "\n\n";

  // compute `A` with soft-thresholded singular values
  A = U.head_cols(minNT) * arma::diagmat(sv) * V.head_cols(minNT).t();

  Rcpp::Rcout << "First row of A = " << A.row(0) << "\n";

  // second update of `a`: `a -= A`
  a -= A;
  Rcpp::Rcout << "First row of a = " << a.row(0) << "\n";

};

// Compute the projection of `A + b` on `B = {RX | diag(X)=0}` and update `b`.
// to do so: solve N regressions of `Y(i)`, with `Y = A + b`, on `R^(-i)`,
// where `Y(i)` is the i-th column of `Y` and `R^(-i)` is `R` without the i-th column,
// then store appropriately the computed regression coefficients
// along the non-diagonal elements of `X`.
void DykstraAP::ComputeStepB() {

  // first update of `b`: `b += A`
  b += A;
  Rcpp::Rcout << "Expectation of b = " << arma::mean(b) << "\n";

  // compute `X` by running N linear regressions of b^(i),
  // which is the i-th column of b, on Ri, which is R without the i-th
  // column
  for (unsigned int i = 0; i < N; ++i) {

    // const arma::uvec columns = join_vert(
    //   assets_idx.head(i),
    //   assets_idx.tail(N-1-i)
    // );

    arma::mat Ri = R;
    Ri.shed_col(i);
    const arma::vec coeff = arma::solve(
      Ri.t() * Ri, //+ lambda * arma::eye(N-1, N-1),
      Ri.t() * b.col(i),
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

  // compute projection `B`
  B = R * X;

  Rcpp::Rcout << "First row of B = " << B.row(0) << "\n";

  // second update of `b`: `b -= B`
  b -= B;

  Rcpp::Rcout << "First row of b = " << b.row(0) << "\n";

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
const arma::rowvec& DykstraAP::GetObjective() const {
  return objective;
};

// get solution
const arma::mat& DykstraAP::GetSolution() const {
  return X;
};
