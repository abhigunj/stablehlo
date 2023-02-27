module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<1> : tensor<2x1xi32>
    %1:2 = call @inputs() : () -> (tensor<3x5x40xi16>, tensor<3x5x2xi16>)
    %2 = call @expected() : () -> tensor<3x5x40xi16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<i16>
      stablehlo.return %5 : tensor<i16>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [2], scatter_dims_to_operand_dims = [2], index_vector_dim = 1>, unique_indices = false} : (tensor<3x5x40xi16>, tensor<2x1xi32>, tensor<3x5x2xi16>) -> tensor<3x5x40xi16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<3x5x40xi16>, tensor<3x5x40xi16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x5x40xi16>, tensor<3x5x2xi16>) {
    %0 = stablehlo.constant dense<"0x01000300FDFF000001000400FDFF0000FDFF000000000300FFFFFDFF02000200FCFF01000000000001000700010000000000FEFFFCFF0000FEFF03000500000000000200FDFFFCFFF9FF0500000002000000040000000000FEFF0100FFFF0200FBFFFDFF00000400FDFFFBFF05000000000003000000FFFFFFFF07000000FEFFFDFF030000000200FFFF0000000000000300FEFF0600FBFF02000000FFFFFCFF0100FBFF00000500FDFF05000200000001000100FEFFFEFF010000000400FEFF010000000000FEFF00000000FFFFFDFF05000200040000000300000000000000FDFF000000000300FBFF00000200000001000100010003000700FEFF0100FFFF01000000FDFFFFFFFFFF050000000300FEFFFBFFFDFF0100000002000300FEFFFCFF0100000002000300FFFF0200FAFFFEFFFFFFFEFFFBFF0000F8FF0200FFFF0000060002000300060004000400FDFFFEFF0200FEFF0200FEFFFEFFFAFF0200000003000000FFFF01000100FEFF030000000000FFFF0200000000000000FCFF0000FEFF0200030002000000010003000000FEFF0000FEFFFEFF0000F9FF00000100000000000000FFFF000002000000FEFF00000300FAFF02000300000001000000FBFFF7FFFCFF0100FEFF0200FFFFFEFF0100FDFFFAFF000001000300FFFF03000200010000000400FDFF010001000100FEFFFFFFFFFF0000FDFFFFFF0200020002000200FFFF01000000FCFF0000000002000000FFFF030006000000FEFFFEFF0700030000000200FBFF0400000002000100030000000500FBFFFEFFFDFF0000020000000200FFFF00000000010000000200FDFF0300FFFFFDFF0500FFFFFEFFFAFFFEFF0300FFFFFEFF020000000000FDFF00000400FEFF04000300FFFFFAFFFCFF0300FDFF000001000000FFFF0400010004000200FEFF00000000FBFF0200FBFF0200FFFFFAFF0300FDFFFFFF09000100FDFF00000000FEFF0200FCFFFFFF0000FDFFFDFF0000FFFF0000030000000100FCFFFDFFFFFFFFFFFCFF000000000000040005000000020000000400FEFFFFFF0200FCFF0300FFFF00000000FFFF0000FDFFFCFF00000300FFFF00000000FFFF00000100FFFF0000FEFF0000FEFF0000FFFF0200FFFFFDFFFFFF01000400FEFFFCFFFCFF00000000000005000100000001000100000000000200FFFF0100020004000100FEFFFEFFFFFFFBFF0000FEFFFFFF00000400FAFFFFFFFCFF020001000200F9FF0400030002000100FEFF0000FEFFFCFF0100000000000200020001000000FCFFFFFF00000200FEFFFFFF000001000300FEFFFDFFFFFF0000FBFF01000400FEFF00000200050002000000FBFFFEFF03000000FBFF0400FFFFFFFF0200FEFF020000000000FEFF04000400FFFF04000300020004000000FDFFFEFFFEFFFDFF03000000FEFFFEFFFEFF0400FCFFFCFF0000FFFFFDFF0000FEFF0000FAFF01000100FEFF0000FDFF000002000100FFFFFEFF020000000000FAFF000001000200FBFF0100000000000200FFFFFDFFFCFF0000000003000300FBFF0100FFFFFDFF0200010008000000030003000000050004000300010003000000FFFFFFFF0300FFFF0000FDFF0300FCFF00000000F6FF0600FAFFFFFF0300FCFFFFFFFFFFFEFF02000100FDFF0300FEFF00000100FDFF0200000007000100FFFF"> : tensor<3x5x40xi16>
    %1 = stablehlo.constant dense<[[[-4, 5], [1, 3], [2, -1], [3, 0], [0, 4]], [[0, 0], [-2, 3], [1, -1], [-4, -1], [0, -2]], [[0, 0], [0, 2], [-3, 0], [-2, 0], [-3, 0]]]> : tensor<3x5x2xi16>
    return %0, %1 : tensor<3x5x40xi16>, tensor<3x5x2xi16>
  }
  func.func private @expected() -> tensor<3x5x40xi16> {
    %0 = stablehlo.constant dense<"0x0100C4FFFDFF000001000400FDFF0000FDFF000000000300FFFFFDFF02000200FCFF01000000000001000700010000000000FEFFFCFF0000FEFF03000500000000000200FDFFFCFFF9FF05000000020000000C0000000000FEFF0100FFFF0200FBFFFDFF00000400FDFFFBFF05000000000003000000FFFFFFFF07000000FEFFFDFF030000000200FFFF0000000000000300FEFF0600FBFF02000000FFFFFCFF01000A0000000500FDFF05000200000001000100FEFFFEFF010000000400FEFF010000000000FEFF00000000FFFFFDFF05000200040000000300000000000000FDFF000000000300FBFF00000200000001000000010003000700FEFF0100FFFF01000000FDFFFFFFFFFF050000000300FEFFFBFFFDFF0100000002000300FEFFFCFF0100000002000300FFFF0200FAFFFEFFFFFFFEFFFBFF0000F8FF0200FFFF0000000002000300060004000400FDFFFEFF0200FEFF0200FEFFFEFFFAFF0200000003000000FFFF01000100FEFF030000000000FFFF0200000000000000FCFF0000FEFF020003000200000001000300000000000000FEFFFEFF0000F9FF00000100000000000000FFFF000002000000FEFF00000300FAFF02000300000001000000FBFFF7FFFCFF0100FEFF0200FFFFFEFF0100FDFFFAFF000001000300FFFF0300F4FF010000000400FDFF010001000100FEFFFFFFFFFF0000FDFFFFFF0200020002000200FFFF01000000FCFF0000000002000000FFFF030006000000FEFFFEFF0700030000000200FBFF040000000200FFFF030000000500FBFFFEFFFDFF0000020000000200FFFF00000000010000000200FDFF0300FFFFFDFF0500FFFFFEFFFAFFFEFF0300FFFFFEFF020000000000FDFF00000400FEFF04000300FFFFFAFFF0FF0300FDFF000001000000FFFF0400010004000200FEFF00000000FBFF0200FBFF0200FFFFFAFF0300FDFFFFFF09000100FDFF00000000FEFF0200FCFFFFFF0000FDFFFDFF0000FFFF0000030000000000FCFFFDFFFFFFFFFFFCFF000000000000040005000000020000000400FEFFFFFF0200FCFF0300FFFF00000000FFFF0000FDFFFCFF00000300FFFF00000000FFFF00000100FFFF0000FEFF0000FEFF0000FFFF0200FFFFFDFFFFFF01000400FEFFFCFFFCFF00000000000005000100000001000100000000000200FFFF0100020004000100FEFFFEFFFFFFFBFF0000FEFFFFFF00000400FAFFFFFFFCFF020000000200F9FF0400030002000100FEFF0000FEFFFCFF0100000000000200020001000000FCFFFFFF00000200FEFFFFFF000001000300FEFFFDFFFFFF0000FBFF01000400FEFF000002000500020000000000FEFF03000000FBFF0400FFFFFFFF0200FEFF020000000000FEFF04000400FFFF04000300020004000000FDFFFEFFFEFFFDFF03000000FEFFFEFFFEFF0400FCFFFCFF0000FFFFFDFF0000FEFF0000000001000100FEFF0000FDFF000002000100FFFFFEFF020000000000FAFF000001000200FBFF0100000000000200FFFFFDFFFCFF0000000003000300FBFF0100FFFFFDFF0200010008000000030003000000050004000300010003000000FFFFFFFF0300FFFF0000FDFF0300FCFF00000000F6FF0600FAFFFFFF0300FCFFFFFFFFFFFEFF02000100FDFF0300FEFF00000100FDFF0200000007000100FFFF"> : tensor<3x5x40xi16>
    return %0 : tensor<3x5x40xi16>
  }
}