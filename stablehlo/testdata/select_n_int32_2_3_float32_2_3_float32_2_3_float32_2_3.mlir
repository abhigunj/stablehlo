// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:4 = call @inputs() : () -> (tensor<2x3xi32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>)
    %1 = call @expected() : () -> tensor<2x3xf32>
    %c = stablehlo.constant dense<1> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x3xi32>
    %3 = stablehlo.compare  LT, %0#0, %2,  SIGNED : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
    %c_0 = stablehlo.constant dense<2> : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x3xi32>
    %5 = stablehlo.compare  LT, %0#0, %4,  SIGNED : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
    %6 = stablehlo.select %5, %0#2, %0#3 : tensor<2x3xi1>, tensor<2x3xf32>
    %7 = stablehlo.select %3, %0#1, %6 : tensor<2x3xi1>, tensor<2x3xf32>
    stablehlo.custom_call @check.expect_close(%7, %1) {has_side_effect = true} : (tensor<2x3xf32>, tensor<2x3xf32>) -> ()
    return %7 : tensor<2x3xf32>
  }
  func.func private @inputs() -> (tensor<2x3xi32> {mhlo.layout_mode = "default"}, tensor<2x3xf32> {mhlo.layout_mode = "default"}, tensor<2x3xf32> {mhlo.layout_mode = "default"}, tensor<2x3xf32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[0, 0, 2], [2, 1, 0]]> : tensor<2x3xi32>
    %cst = stablehlo.constant dense<[[3.41892362, -0.968371093, -3.74343753], [-1.74075079, -6.24598694, 1.09240973]]> : tensor<2x3xf32>
    %cst_0 = stablehlo.constant dense<[[5.66154289, -2.86463428, 0.038875252], [1.28850126, 0.555465043, 4.92511129]]> : tensor<2x3xf32>
    %cst_1 = stablehlo.constant dense<[[2.02817249, -5.940207, -2.58250499], [3.40481019, 5.68319035, 0.296268404]]> : tensor<2x3xf32>
    return %c, %cst, %cst_0, %cst_1 : tensor<2x3xi32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>
  }
  func.func private @expected() -> (tensor<2x3xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[3.41892362, -0.968371093, -2.58250499], [3.40481019, 0.555465043, 1.09240973]]> : tensor<2x3xf32>
    return %cst : tensor<2x3xf32>
  }
}