// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<7x3x4xui32>, tensor<7x4xui32>)
    %1 = call @expected() : () -> tensor<7x3xui32>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>, precision_config = [#stablehlo<precision HIGH>, #stablehlo<precision HIGH>]} : (tensor<7x3x4xui32>, tensor<7x4xui32>) -> tensor<7x3xui32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<7x3xui32>, tensor<7x3xui32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<7x3x4xui32>, tensor<7x4xui32>) {
    %0 = stablehlo.constant dense<[[[4, 1, 0, 3], [4, 3, 4, 0], [1, 0, 0, 2]], [[3, 2, 4, 3], [2, 2, 1, 1], [0, 1, 1, 1]], [[0, 2, 2, 0], [5, 1, 2, 0], [5, 3, 4, 2]], [[5, 1, 1, 3], [1, 0, 4, 0], [0, 2, 3, 1]], [[0, 2, 2, 0], [0, 0, 3, 2], [0, 2, 0, 2]], [[3, 2, 2, 3], [0, 0, 3, 4], [0, 3, 1, 2]], [[3, 0, 7, 0], [2, 3, 0, 2], [0, 0, 1, 1]]]> : tensor<7x3x4xui32>
    %1 = stablehlo.constant dense<[[1, 0, 1, 2], [3, 0, 1, 0], [0, 0, 0, 2], [0, 2, 0, 5], [2, 0, 3, 2], [0, 0, 5, 8], [1, 1, 3, 2]]> : tensor<7x4xui32>
    return %0, %1 : tensor<7x3x4xui32>, tensor<7x4xui32>
  }
  func.func private @expected() -> tensor<7x3xui32> {
    %0 = stablehlo.constant dense<[[10, 8, 5], [13, 7, 1], [0, 0, 4], [17, 0, 9], [6, 13, 4], [34, 47, 21], [24, 9, 5]]> : tensor<7x3xui32>
    return %0 : tensor<7x3xui32>
  }
}