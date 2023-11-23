// Author: Alberto Quaini

#ifndef SIMPLEXPROJ_H
#define SIMPLEXPROJ_H

#include <RcppArmadillo.h>

// Function for internal use:
// `ProjectOntoSimplex` projects the `input` vector onto the simplex of radius `tau`.
// Algorithm adapted from: Chen, Yunmei, and Xiaojing Ye.
// "Projection onto a simplex." arXiv preprint arXiv:1101.6081 (2011).
void ProjectOntoSimplex(arma::vec& input, const double radius);

#endif
