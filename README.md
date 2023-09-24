
# markovitzRRR: Markovitz optimal portfolio via Reduced Rank Regression

<!-- badges: start -->
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
<!-- badges: end -->

Author: Alberto Quaini

Efficient implementation of Markovitz optimal portfolio via Reduced Rank Regression

## Installation

### Building from source

To install the latest (possibly unstable) development version from
GitHub, you can pull this repository and install it from the `R` command
line via

```R
# if you already have package `devtools` installed, you can skip the next line
install.packages("devtools")
devtools::install_github("a91quaini/markovitzRRR")
```

Package `markovitzRRR` contains `C++` code that needs to be
compiled, so you may need to download and install the [necessary tools
for MacOS](https://cran.r-project.org/bin/macosx/tools/) or the
[necessary tools for
Windows](https://cran.r-project.org/bin/windows/Rtools/).


## Example

This is a basic example which shows you how to solve a common problem:

``` r
library(markovitzRRR)
## basic example code
```

