test_that("Test MarkovitzRRR", {

  # simulate returns
  n_assets = 5
  returns = markovitzRRR::returns[1:20, 2:(n_assets + 1)]
  max_iter = 5

  lambda = 0.2

  expect_no_error(
    MarkovitzRRR(
      returns,
      lambda,
      max_iter = max_iter
    )
  )

  expect_no_error(
    MarkovitzRRR(
      returns,
      lambda,
      max_iter = 5
    )
  )

  expect_no_error(
    MarkovitzRRR(
      returns,
      lambda,
      objective_type = 'a',
      max_iter = 5
    )
  )

  expect_no_error(
    MarkovitzRRR(
      returns,
      lambda,
      penalty_type = 'a',
      max_iter = max_iter
    )
  )

  expect_no_error(
    MarkovitzRRR(
      returns,
      lambda,
      step_size_type = 'c',
      max_iter = max_iter
    )
  )

  expect_no_error(
    MarkovitzRRR(
      returns,
      lambda,
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
      lambda,
      max_iter = max_iter
    )$solution,
    n_assets * n_assets
  )

  expect_error(
    MarkovitzRRR(
      returns = "s",
      lambda
    )
  )

  expect_error(
    MarkovitzRRR(
      returns,
      lambda = -0.3
    )
  )

  expect_error(
    MarkovitzRRR(
      returns,
      lambda = "s"
    )
  )

  expect_error(
    MarkovitzRRR(
      returns,
      lambda,
      objective_type = c(2, 3)
    )
  )

  expect_error(
    MarkovitzRRR(
      returns,
      lambda,
      penalty_type = c(2, 3)
    )
  )

  expect_error(
    MarkovitzRRR(
      returns,
      lambda,
      step_size_type = c(2, 3)
    )
  )

  expect_error(
    MarkovitzRRR(
      returns,
      lambda,
      step_size_constant = "s"
    )
  )

  expect_error(
    MarkovitzRRR(
      returns,
      lambda,
      max_iter = "s"
    )
  )

  # test the subgradient computation method used when N > 0.9 T
  n_assets = 19
  returns = markovitzRRR::returns[1:20, 2:(n_assets + 1)]
  expect_no_error(
    MarkovitzRRR(
      returns,
      lambda,
      max_iter = max_iter
    )
  )

  expect_error(
    MarkovitzRRR(
      returns,
      lambda,
      tolerance = "s"
    )
  )

})
