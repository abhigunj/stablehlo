module {
  func.func @add(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>) -> tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>> {
    %0 = stablehlo.uniform_quantize %arg0 : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
    %1 = stablehlo.uniform_dequantize %0 : (tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>) -> tensor<16x16xf32>
    //%2 = stablehlo.uniform_quantize %arg1 : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
    %3 = stablehlo.uniform_dequantize %arg1 : (tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>) -> tensor<16x16xf32>
    %4 = stablehlo.add %1, %3 : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
    %5 = stablehlo.uniform_quantize %4 : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
    func.return %5 : tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
  }
}

//func.func @add(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>) -> tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>> {
//  %0 = stablehlo.uniform_quantize %arg0 : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>
//  %1 = stablehlo.uniform_dequantize %0 : (tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>) -> tensor<16x16xf32>
//  %2 = stablehlo.uniform_dequantize %arg1 : (tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>) -> tensor<16x16xf32>
//  %3 = stablehlo.add %1, %2 : tensor<16x16xf32>
//  %4 = stablehlo.add %0, %arg1 : tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>
//  %5 = stablehlo.uniform_quantize %3 : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>
//  return %4 : tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>
//}
