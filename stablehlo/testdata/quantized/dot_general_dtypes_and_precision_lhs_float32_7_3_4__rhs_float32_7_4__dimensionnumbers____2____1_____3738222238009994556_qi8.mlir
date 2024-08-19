module attributes {jax.uses_shape_polymorphism = true} {
  func.func @main() -> tensor<i1> {
    %cst = stablehlo.constant dense<[[[-5.06569719, -6.63685608, 1.26659703, -2.23289418], [0.328246891, -3.34304667, 1.73156273, 1.53923786], [3.4846642, 0.705635368, -2.22549462, 0.242756724]], [[-1.70394194, -2.51021576, -2.09778261, 4.19714069], [3.01307082, -2.76771045, -1.0915395, -1.73421705], [0.299419522, 0.845451295, -0.819909095, 3.19720507]], [[1.99056602, -1.93757975, 2.53377318, -3.09514332], [1.75113702, -2.4899106, -0.79994142, -2.40296936], [-0.0599169321, -0.23335208, 2.01964235, -0.321210414]], [[0.842348873, -0.998543977, -0.127073184, -1.44472122], [-1.84733689, -4.396110e+00, -2.45703483, -2.336200e+00], [-0.861453115, -2.29369187, 0.675022483, -3.52336645]], [[-0.564810574, -5.08944082, 1.16774499, -1.53107715], [1.45824754, -5.4955616, 0.416088909, -3.83327866], [2.1967895, 1.75280392, -0.537377656, -0.707910239]], [[-8.01627826, 4.58239603, 5.02564526, 0.80679822], [-0.836623311, 1.24237132, 2.776280e+00, 0.584487915], [-0.0894802064, 4.31166935, -0.373315901, -0.898094534]], [[-2.55243015, 7.37620639, 1.54967582, -0.28037253], [-0.838343977, 3.83477807, 1.47972381, -0.201482028], [0.0161808729, 4.45749187, -1.6296674, -0.955628037]]]> : tensor<7x3x4xf32>
    %cst_0 = stablehlo.constant dense<[[-0.55160737, -0.90700668, 1.06096828, 5.65155697], [-1.60879683, -1.79970551, -3.59757376, -1.44012725], [0.368852973, -3.22422409, -3.70365787, -3.66013098], [2.05148673, 0.567722142, -8.35568714, -1.18737185], [-5.40125656, 2.91858554, 0.214062393, -5.193070e-01], [-0.997602939, -6.650330e-02, 1.78420949, -1.0177536], [-0.352346808, -2.6139946, -5.54016876, -2.50792646]]> : tensor<7x4xf32>
    %cst_1 = stablehlo.constant dense<[[1.00496554, 1.99946284, 2.407730e-01], [0.000000e+00, 0.000000e+00, 0.000000e+00], [0.366393715, 0.366393715, 0.000000e+00], [0.84793973, 0.000000e+00, 0.000000e+00], [0.21983622, 0.0942155272, 1.00496554], [1.00496554, 1.00496554, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00]]> : tensor<7x3xf32>
    %0 = stablehlo.uniform_quantize %cst_0 : (tensor<7x4xf32>) -> tensor<7x4x!quant.uniform<i8:f32, 0.0039208187776453357:-128>>
    %1 = stablehlo.uniform_quantize %cst : (tensor<7x3x4xf32>) -> tensor<7x3x4x!quant.uniform<i8:f32, 0.0039215482917486456:-128>>
    %2 = stablehlo.dot_general %1, %0, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [HIGHEST, HIGHEST] : (tensor<7x3x4x!quant.uniform<i8:f32, 0.0039215482917486456:-128>>, tensor<7x4x!quant.uniform<i8:f32, 0.0039208187776453357:-128>>) -> tensor<7x3x!quant.uniform<i32:f32, 1.5375680179731079E-5>>
    %3 = stablehlo.uniform_quantize %2 : (tensor<7x3x!quant.uniform<i32:f32, 1.5375680179731079E-5>>) -> tensor<7x3x!quant.uniform<i8:f32, 0.010468391343659046:-128>>
    %4 = stablehlo.uniform_dequantize %3 : (tensor<7x3x!quant.uniform<i8:f32, 0.010468391343659046:-128>>) -> tensor<7x3xf32>
    %5 = stablehlo.custom_call @check.eq(%cst_1, %4) : (tensor<7x3xf32>, tensor<7x3xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
}
