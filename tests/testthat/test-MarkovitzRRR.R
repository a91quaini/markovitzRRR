test_that("Test MarkovitzRRR", {

  # simulate returns
  n_assets = 5
  returns = markovitzRRR::returns[1:20, 2:(n_assets + 1)]
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
  returns = markovitzRRR::returns[1:20, 2:(n_assets + 1)]
  expect_no_error(
    MarkovitzRRR(
      returns,
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

})
