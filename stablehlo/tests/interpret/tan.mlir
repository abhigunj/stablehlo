// RUN: stablehlo-translate --interpret -split-input-file %s

// -----

func.func @tan_op_test_c64() {
  %0 = stablehlo.constant dense<[(1.5, 2.5), (3.5, 4.5)]> : tensor<2xcomplex<f64>>
  %1 = stablehlo.tan %0 : tensor<2xcomplex<f64>>
  check.expect_almost_eq_const %1, dense<[(0.0019273435237456358, 1.0134287782038933), (1.6212700415590609E-4, 0.99981392630805066)]> : tensor<2xcomplex<f64>>
  func.return
}
