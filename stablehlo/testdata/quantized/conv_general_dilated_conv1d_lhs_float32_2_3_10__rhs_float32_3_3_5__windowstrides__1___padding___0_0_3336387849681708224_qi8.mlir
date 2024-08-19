module attributes {jax.uses_shape_polymorphism = true} {
  func.func @main() -> tensor<i1> {
    %cst = stablehlo.constant dense<[[[-2.39628553, -2.61640573, -1.71100402, 0.754390299, 4.43838644, 0.543790221, -4.81503296, 2.85119772, 0.0272705927, -1.70016646], [3.82113361, -0.536128283, -2.83340192, -3.46321607, 1.16630661, -2.06615543, 0.120180897, -0.32935223, 2.26144934, -2.39108396], [-1.33467293, 0.329310775, 1.34398723, 2.51446867, 2.428520e+00, 3.4680829, -3.10366297, 3.0266571, 2.62947488, -0.478714615]], [[-4.57713127, -3.58981895, 1.54749513, 0.514547884, 2.09321499, -0.390078187, -0.13165769, 5.04404354, 0.0416141041, 0.771804571], [3.50332165, 4.41446114, 5.62385559, -5.87154913, -4.40485573, 6.00450706, 6.13054466, 8.6186285, -0.558318675, -4.69300604], [-0.848347127, -0.702094197, -1.04472029, -0.052434817, -1.83562136, -3.88773799, 3.57837367, -2.34969687, 1.86881018, -1.13490331]]]> : tensor<2x3x10xf32>
    %cst_0 = stablehlo.constant dense<[[[7.660650e-01, 3.57861495, 6.736780e+00, -3.81931186, -3.55869794], [-3.55001354, 0.11664889, 4.946780e+00, -2.04585624, -5.01829767], [1.09930515, 3.48784447, 3.81765532, -3.41729975, 2.87953019]], [[1.07684827, 2.18463969, -1.48767233, 1.09800053, 0.996845901], [2.23116779, -1.94385302, 1.27256846, -0.673200488, -1.19993186], [0.76928246, 3.64450717, 0.108128168, -1.11739159, 1.73053598]], [[5.4585433, 0.812036573, -1.19025648, -0.801120817, -1.12692165], [0.858582735, -0.642224252, 3.22822118, 0.577595055, -0.287795871], [-1.0398047, -0.445783377, -3.34618831, 2.61982918, 1.01340723]]]> : tensor<3x3x5xf32>
    %cst_1 = stablehlo.constant dense<[[[2.34015727, 4.08227444, 5.74638605, 6.24041939, 4.42029715, 3.43223071], [4.1862812, 3.90026212, 4.1862812, 5.61637735, 6.47443533, 1.45609784], [2.8601923, 2.57417297, 2.60017467, 2.6261766, 4.42029715, 2.13214326]], [[2.10614157, 1.63811016, 3.27622032, 2.39216089, 3.87426043, 3.12020969], [3.51023602, 2.990201, 3.51023602, 3.51023602, 4.16027975, 3.8222568], [1.84612405, 1.66411185, 3.84825873, 3.90026212, 3.5882411, 2.8601923]]]> : tensor<2x3x6xf32>
    %0 = stablehlo.uniform_quantize %cst_0 : (tensor<3x3x5xf32>) -> tensor<3x3x5x!quant.uniform<i8:f32, 0.0039215482917486456:-128>>
    %1 = stablehlo.uniform_quantize %cst : (tensor<2x3x10xf32>) -> tensor<2x3x10x!quant.uniform<i8:f32, 0.0039212498010373579:-128>>
    %2 = stablehlo.convolution(%1, %0) dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<2x3x10x!quant.uniform<i8:f32, 0.0039212498010373579:-128>>, tensor<3x3x5x!quant.uniform<i8:f32, 0.0039215482917486456:-128>>) -> tensor<2x3x6x!quant.uniform<i32:f32, 1.5377370458777768E-5>>
    %3 = stablehlo.uniform_quantize %2 : (tensor<2x3x6x!quant.uniform<i32:f32, 1.5377370458777768E-5>>) -> tensor<2x3x6x!quant.uniform<i8:f32, 0.026001746981751686:-128>>
    %4 = stablehlo.uniform_dequantize %3 : (tensor<2x3x6x!quant.uniform<i8:f32, 0.026001746981751686:-128>>) -> tensor<2x3x6xf32>
    %5 = stablehlo.custom_call @check.eq(%cst_1, %4) : (tensor<2x3x6xf32>, tensor<2x3x6xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
}
