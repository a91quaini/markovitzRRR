test_that("Test MarkovitzRRR", {

  # simulate returns
  n_assets = 3
  returns = markovitzRRR::returns[1:30, 2:(n_assets + 1)]
  max_iter = 5

  lambda1 = 0.2
  lambda2 = 0.2

  expect_no_error(
    MarkovitzRRR(
      returns,
      lambda1,
      lambda2,
      max_iter = max_iter
    )
  )

  expect_no_error(
    MarkovitzRRR(
      returns,
      lambda1,
      lambda2,
      max_iter = 5
    )
  )

  expect_no_error(
    MarkovitzRRR(
      returns,
      lambda1,
      lambda2,
      penalty_type = 'a',
      max_iter = max_iter
    )
  )

  expect_no_error(
    MarkovitzRRR(
      returns,
      lambda1,
      lambda2,
      step_size_type = 'c',
      max_iter = max_iter
    )
  )

  expect_no_error(
    MarkovitzRRR(
      returns,
      lambda1,
      lambda2,
      step_size_type = 'p',
      max_iter = max_iter
    )
  )

  # expect_equal(
  #   MarkovitzRRR(
  #     returns,
  #     lambda,
  #     tolerance = 1.e-6
  #   )$status,
  #   "solved"
  # )

  expect_length(
    MarkovitzRRR(
      returns,
      lambda1,
      lambda2,
      max_iter = max_iter
    )$solution,
    n_assets * n_assets
  )

  expect_error(
    MarkovitzRRR(
      returns = "s",
      lambda1,
      lambda2,
    )
  )

  expect_error(
    MarkovitzRRR(
      returns,
      lambda1 = -0.3
    )
  )

  expect_error(
    MarkovitzRRR(
      returns,
      lambda2 = -0.3
    )
  )

  expect_error(
    MarkovitzRRR(
      returns,
      lambda1,
      lambda2,
      initial_solution = "c"
    )
  )

  expect_error(
    MarkovitzRRR(
      returns,
      lambda1 = "s",
      lambda2
    )
  )

  expect_error(
    MarkovitzRRR(
      returns,
      lambda1,
      lambda2 = "s"
    )
  )

  expect_error(
    MarkovitzRRR(
      returns,
      lambda1,
      lambda2,
      objective_type = c(2, 3)
    )
  )

  expect_error(
    MarkovitzRRR(
      returns,
      lambda1,
      lambda2,
      penalty_type = c(2, 3)
    )
  )

  expect_error(
    MarkovitzRRR(
      returns,
      lambda1,
      lambda2,
      step_size_type = c(2, 3)
    )
  )

  expect_error(
    MarkovitzRRR(
      returns,
      lambda1,
      lambda2,
      step_size_constant = "s"
    )
  )

  expect_error(
    MarkovitzRRR(
      returns,
      lambda1,
      lambda2,
      max_iter = "s"
    )
  )

  # test the subgradient computation method used when N > 0.9 T
  n_assets = 19
  returns_high = markovitzRRR::returns[1:20, 2:(n_assets + 1)]
  expect_no_error(
    MarkovitzRRR(
      returns_high,
      lambda1,
      lambda2,
      max_iter = max_iter
    )
  )

  expect_error(
    MarkovitzRRR(
      returns,
      lambda1,
      lambda2,
      tolerance = "s"
    )
  )

  expect_no_error(
    MarkovitzRRR(
      returns,
      lambda1 = 1000,
      lambda2 = 1000,
      max_iter = max_iter
    )
  )

  result <- MarkovitzRRR(
    returns,
    lambda1,
    lambda2,
    max_iter = max_iter
  )
  expect_true(is.list(result))
  expect_true(all(c("solution", "objective", "weights") %in% names(result)))

  expect_error(
    MarkovitzRRR(
      returns = "invalid",
      lambda1 = 0.1,
      lambda2 = 0.1
    )
  )

  expect_error(
    MarkovitzRRR(
      returns,
      lambda1 = -0.1,
      lambda2 = 0.1
    )
  )
  expect_error(
    MarkovitzRRR(
      returns,
      lambda1 = 0.1,
      lambda2 = -0.1
    )
  )

  expect_error(
    MarkovitzRRR(
      returns,
      lambda1 = 0.1,
      lambda2 = 0.1,
      penalty_type = 3.4
    )
  )
  expect_error(
    MarkovitzRRR(
      returns,
      lambda1 = 0.1,
      lambda2 = 0.1,
      step_size_type = 3.4
    )
  )

  expect_no_error(
    MarkovitzRRR(
      returns,
      lambda1 = 1000,
      lambda2 = 1000
    )
  )
  expect_no_error(
    MarkovitzRRR(
      returns,
      lambda1 = 1e-10,
      lambda2 = 1e-10
    )
  )

  expect_no_error(
    MarkovitzRRR(
      returns,
      lambda1 = 0.1,
      lambda2 = 0.1,
      max_iter = 1
    )
  )
  expect_no_error(
    MarkovitzRRR(
      returns,
      lambda1 = 0.1,
      lambda2 = 0.1,
      tolerance = 1e6
    )
  )

  result <- MarkovitzRRR(
    returns,
    lambda1 = 0.1,
    lambda2 = 0.1
  )
  expect_true(is.matrix(result$solution))
  expect_equal(dim(result$solution), c(ncol(returns), ncol(returns)))

  result1 <- MarkovitzRRR(
    returns,
    lambda1 = 0.1,
    lambda2 = 0.1
  )
  result2 <- MarkovitzRRR(
    returns,
    lambda1 = 0.1,
    lambda2 = 0.1
  )
  expect_equal(result1$weights, result2$weights)

  step_types <- c('d', 's', 'l', 'p', 'c')
  for (step_type in step_types) {
    expect_no_error(
      MarkovitzRRR(
        returns,
        lambda1 = 0.1,
        lambda2 = 0.1,
        step_size_type = step_type
      )
    )
  }

  singular_returns <- matrix(rep(1, 100), ncol = 10)
  expect_no_error(
    MarkovitzRRR(
      singular_returns,
      lambda1 = 0.1,
      lambda2 = 0.1
    )
  )

  expect_no_error(
    MarkovitzRRR(
      returns,
      lambda1 = 0,
      lambda2 = 0
    )
  )

})
