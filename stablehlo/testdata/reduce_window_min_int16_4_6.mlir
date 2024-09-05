// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x5xi16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<4x6xi16>
    %1 = call @expected() : () -> tensor<3x5xi16>
    %c = stablehlo.constant dense<32767> : tensor<i16>
    %2 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i16>) -> tensor<i16>
    %3 = "stablehlo.reduce_window"(%0, %2) <{window_dimensions = array<i64: 2, 2>}> ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      %4 = stablehlo.minimum %arg0, %arg1 : tensor<i16>
      stablehlo.return %4 : tensor<i16>
    }) : (tensor<4x6xi16>, tensor<i16>) -> tensor<3x5xi16>
    stablehlo.custom_call @check.expect_eq(%3, %1) {has_side_effect = true} : (tensor<3x5xi16>, tensor<3x5xi16>) -> ()
    return %3 : tensor<3x5xi16>
  }
  func.func private @inputs() -> (tensor<4x6xi16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[0, 0, 0, -1, 4, 2], [-1, -1, 3, -4, -2, 2], [0, -2, 4, -3, 0, -1], [0, 0, 3, 0, 0, 2]]> : tensor<4x6xi16>
    return %c : tensor<4x6xi16>
  }
  func.func private @expected() -> (tensor<3x5xi16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[-1, -1, -4, -4, -2], [-2, -2, -4, -4, -2], [-2, -2, -3, -3, -1]]> : tensor<3x5xi16>
    return %c : tensor<3x5xi16>
  }
}