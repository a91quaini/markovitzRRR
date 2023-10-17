// Author: Alberto Quaini

#include "simplexproj.h"

void ProjectOntoSimplex(arma::vec& input, const double radius) {

  int dimension = input.n_elem;
  bool found = false;

  arma::vec sorted = arma::sort(input, "descend");

  // vectorized
  // const arma::vec cumsum = arma::cumsum(sorted);
  // const arma::vec thresholds = (
  //   cumsum.head(dimension - 1) - radius
  // ) / arma::regspace(1., dimension - 1.);
  // const arma::Col<bool> found = thresholds >= sorted.tail(dimension - 1);

  double sum = 0.;
  double threshold;

  for (int idx = 0; idx < dimension - 1; ++idx) {

    sum += sorted(idx);

    threshold = (sum - radius) / (idx + 1.);

    if (threshold >= sorted(idx + 1)) {

      found = true;
      break;

    }

  }

  if (!found) {

    threshold = (sum + sorted(dimension - 1) - radius) / dimension;

  }

  input -= threshold;
  input.clamp(0., arma::datum::inf);

}
