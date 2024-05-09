// FUN: sstablehlo-translate --interpret -split-input-file %s

// %operand: [
//            [1, 2, 3]
//           ]
func.func @dynamic_broadcast_in_dim(%operand: tensor<1x3xi64>) -> tensor<2x3x2xi64> {
   %output_dimensions = stablehlo.constant dense<[2, 3, 2]> : tensor<3xi64>
   %result = "stablehlo.dynamic_broadcast_in_dim"(%operand, %output_dimensions) {
      broadcast_dimensions = array<i64: 2, 1>,
      known_expanding_dimensions = array<i64: 0>,
      known_non_expanding_dimensions = array<i64: 1>
    } : (tensor<1x3xi64>, tensor<3xi64>) -> tensor<2x3x2xi64>
    func.return %result : tensor<2x3x2xi64>
 }
// %result: [
//            [
//             [1, 1],
//             [2, 2],
//             [3, 3]
//            ],
//            [
//             [1, 1],
//             [2, 2],
//             [3, 3]
//            ]
//          ]


