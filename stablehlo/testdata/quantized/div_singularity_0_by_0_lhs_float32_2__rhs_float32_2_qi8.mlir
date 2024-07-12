// RUN: stablehlo-translate --interpret -split-input-file %s

module attributes {jax.uses_shape_polymorphism = true} {
  func.func @main() -> tensor<i1> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
    %cst_0 = stablehlo.constant dense<0.501960814> : tensor<2xf32>
    %0 = stablehlo.uniform_quantize %cst : (tensor<2xf32>) -> tensor<2x!quant.uniform<i8:f32, 0.0039059886745378084:-128>>
    %1 = stablehlo.divide %0, %0 : (tensor<2x!quant.uniform<i8:f32, 0.0039059886745378084:-128>>, tensor<2x!quant.uniform<i8:f32, 0.0039059886745378084:-128>>) -> tensor<2x!quant.uniform<i8:f32, 0.0039215686274509803:-128>>
    %2 = stablehlo.uniform_dequantize %1 : (tensor<2x!quant.uniform<i8:f32, 0.0039215686274509803:-128>>) -> tensor<2xf32>
    %3 = stablehlo.custom_call @check.eq(%cst_0, %2) : (tensor<2xf32>, tensor<2xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
}
