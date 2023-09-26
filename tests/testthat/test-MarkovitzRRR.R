test_that("Test MarkovitzRRR", {

  # simulate returns
  n_assets = 5
  returns = markovitzRRR::returns[1:20, 2:(n_assets + 1)]

  lambda = 0.2

  expect_no_error(
    MarkovitzRRR(
      returns,
      lambda,
      max_iter = 100
    )
  )

  expect_no_error(
    MarkovitzRRR(
      returns,
      lambda,
      step_size_type = 'c'
    )
  )

  expect_no_error(
    MarkovitzRRR(
      returns,
      lambda,
      step_size_type = 'p'
    )
  )

  expect_equal(
    MarkovitzRRR(
      returns,
      lambda,
      tolerance = 1.e-6
    )$status,
    "solved"
  )

  expect_length(
    MarkovitzRRR(
      returns,
      lambda
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
      lambda = "s"
    )
  )

  expect_error(
    MarkovitzRRR(
      returns,
      lambda,
      max_iter = "s"
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
      tolerance = "s"
    )
  )

})
