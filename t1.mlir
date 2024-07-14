module @cross_replica_variadic_inputs {
  func.func @all_gather(%arg0 : tensor<2x2xi64>, %arg1 : tensor<2x2xi64>) -> (tensor<2x4xi64>, tensor<2x4xi64>) {
    %result:2 = "stablehlo.all_gather"(%arg0, %arg1) {
      all_gather_dim = 1 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
    } : (tensor<2x2xi64>, tensor<2x2xi64>) -> (tensor<2x4xi64>, tensor<2x4xi64>)
    return %result#0, %result#1 : tensor<2x4xi64>, tensor<2x4xi64>
  }
  func.func @main() {
    %process0_operand0 = stablehlo.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>
    %process0_operand1 = stablehlo.constant dense<[[5, 6], [7, 8]]> : tensor<2x2xi64>
    %process1_operand0 = stablehlo.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>
    %process1_operand1 = stablehlo.constant dense<[[5, 6], [7, 18]]> : tensor<2x2xi64>
    %results:4 = "interpreter.run_parallel"(%process0_operand0, %process1_operand0, %process0_operand1, %process1_operand1) {
      programs=[[@all_gather], [@all_gather]]
    } : (tensor<2x2xi64>, tensor<2x2xi64>, tensor<2x2xi64>, tensor<2x2xi64>) -> (tensor<2x4xi64>, tensor<2x4xi64>, tensor<2x4xi64>, tensor<2x4xi64>)
    check.expect_eq_const %results#0, dense<[[1, 2, 5, 6],
                                             [3, 4, 7, 8]]> : tensor<2x4xi64>
    check.expect_eq_const %results#1, dense<[[1, 2, 5, 6],
                                             [3, 4, 7, 8]]> : tensor<2x4xi64>
    check.expect_eq_const %results#2, dense<[[1, 2, 5, 6],
                                             [3, 4, 7, 8]]> : tensor<2x4xi64>
    check.expect_eq_const %results#3, dense<[[1, 2, 5, 6],
                                             [3, 4, 7, 18]]> : tensor<2x4xi64>
    func.return
  }
}