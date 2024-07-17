module @cross_replica {
  func.func @all_reduce(%operand0 : tensor<4xi64>, %operand1 : tensor<4xi64>) -> (tensor<4xi64>, tensor<4xi64>) {
    %result:2 = "stablehlo.all_reduce"(%operand0, %operand1) ({
      ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
        %0 = stablehlo.add %arg0, %arg1 : tensor<i64>
        stablehlo.return %0 : tensor<i64>
    }) {
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
    } : (tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>, tensor<4xi64>)
    return %result#0, %result#1 : tensor<4xi64>, tensor<4xi64>
  }
  func.func @main() -> (tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) {
    %inputs0_0 = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi64>
    %inputs0_1 = stablehlo.constant dense<[5, 6, 7, 8]> : tensor<4xi64>
    %inputs1_0 = stablehlo.constant dense<[11, 12, 13, 14]> : tensor<4xi64>
    %inputs1_1 = stablehlo.constant dense<[15, 16, 17, 18]> : tensor<4xi64>
    %results:4 = "interpreter.run_parallel"(%inputs0_0, %inputs1_0, %inputs0_1, %inputs1_1) {
      programs=[[@all_reduce], [@all_reduce]]
    } : (tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>)
    //check.expect_eq_const %results#0, dense<[6, 8, 10, 12]> : tensor<4xi64>
    //check.expect_eq_const %results#1, dense<[6, 8, 10, 12]> : tensor<4xi64>
    //check.expect_eq_const %results#0, dense<[6, 8, 10, 22]> : tensor<4xi64>
    //check.expect_eq_const %results#1, dense<[6, 8, 10, 22]> : tensor<4xi64>
    func.return %results#0, %results#1, %results#2, %results#3 : tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>
  }
}