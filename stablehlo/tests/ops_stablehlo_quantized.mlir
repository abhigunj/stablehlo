// RUN: stablehlo-opt %s -verify-diagnostics -split-input-file -allow-unregistered-dialect | FileCheck %s


// -----
// Tests for StableHLO OPs supporting per-axis quantization. These OPs also support per-tensor quantization.

// CHECK-LABEL: @ops_per_axis_quantization
func.func @ops_per_axis_quantization(
  %arg0: tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:2, {0.1:-30, 0.5:-20}>>,
  %arg1: tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:0, {0.1:-30}>>,
  %token0: !stablehlo.token) {
  %bitcast_convert = "stablehlo.bitcast_convert"(%arg0) : (tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:2, {0.1:-30, 0.5:-20}>>) -> tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:2, {0.1:-30, 0.5:-20}>>
  %broadcast_in_dim_1 = "stablehlo.broadcast_in_dim" (%arg0) {broadcast_dimensions = array<i64: 0, 1, 3>} : (tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:2, {0.1:-30, 0.5:-20}>>) -> tensor<1x2x3x2x!quant.uniform<i8<-128:127>:f32:3, {0.1:-30, 0.5:-20}>>
  %broadcast_in_dim_2 = "stablehlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = array<i64: 0, 1, 2>} : (tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:0, {0.1:-30}>>) -> tensor<2x2x2x!quant.uniform<i8<-128:127>:f32:0, {0.1:-30, 0.1:-30}>>
  %custom_call = "stablehlo.custom_call" (%arg0) {call_target_name = "foo"} : (tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:2, {0.1:-30, 0.5:-20}>>) -> tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:2, {0.1:-30, 0.5:-20}>>
  %outfeed = "stablehlo.outfeed"(%arg0, %token0) {outfeed_config = ""} : (tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:2, {0.1:-30, 0.5:-20}>>, !stablehlo.token) -> !stablehlo.token
  %reshape = "stablehlo.reshape" (%arg0) : (tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:2, {0.1:-30, 0.5:-20}>>) -> tensor<2x2x!quant.uniform<i8<-128:127>:f32:1, {0.1:-30, 0.5:-20}>>
  %send = "stablehlo.send"(%arg0, %token0) {channel_handle = #stablehlo.channel_handle<handle = 5, type = 2>, is_host_transfer = true} : (tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:2, {0.1:-30, 0.5:-20}>>, !stablehlo.token) -> !stablehlo.token
  %transpose = "stablehlo.transpose"(%arg0) {permutation = array<i64: 0, 2, 1>}: (tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:2, {0.1:-30, 0.5:-20}>>) -> tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:1, {0.1:-30, 0.5:-20}>>
  %tuple = "stablehlo.tuple"(%arg1, %arg1) : (tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:0, {0.1:-30}>>, tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:0, {0.1:-30}>>) -> tuple<tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:0, {0.1:-30}>>, tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:0, {0.1:-30}>>>
  %uniform_dequantize = "stablehlo.uniform_dequantize" (%arg0) : (tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:2, {0.1:-30, 0.5:-20}>>) -> tensor<1x2x2xf32>
  %uniform_quantize = "stablehlo.uniform_quantize" (%arg0) : (tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:2, {0.1:-30, 0.5:-20}>>) -> tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:2, {0.1:-30, 0.5:-20}>>
  func.return
}

// %arg1 can be a per-axis Quantized
func.func @dot_general_per_axis_quantization(%arg0: tensor<2x3x4x!quant.uniform<i8:f32, 1.0:17>>, %arg1: tensor<2x3x5x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<2x4x5x!quant.uniform<i8:f32:0, {0.1:-30}>> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<2x3x4x!quant.uniform<i8:f32, 1.0:17>>, tensor<2x3x5x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<2x4x5x!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return %0 : tensor<2x4x5x!quant.uniform<i8:f32:0, {0.1:-30}>>
}

// -----
// Tests for StableHLO OPs supporting per-tensor quantization. These OPs may or may not support per-axis quantization

func.func @ops_per_tensor_quantization(
  %arg0: tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>,
  %arg1: tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>,
  %arg2: tensor<!quant.uniform<i8:f32, 1.0:17>>,
  %arg3: tensor<2x4x!quant.uniform<i8:f32, 1.0:17>>,
  %token0: !stablehlo.token) {
  %abs = "stablehlo.abs"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %add = "stablehlo.add"(%arg0, %arg1) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %all_gather = "stablehlo.all_gather"(%arg3) { all_gather_dim = 1 : i64, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<2x4x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<2x4x!quant.uniform<i8:f32, 1.0:17>>
  %all_to_all = "stablehlo.all_to_all"(%arg3) { split_dimension = 1 : i64, concat_dimension = 1 : i64, split_count = 2 : i64, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>} : (tensor<2x4x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<2x4x!quant.uniform<i8:f32, 1.0:17>>
  %atan2 = "stablehlo.atan2"(%arg0, %arg1) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %bitcast_convert = "stablehlo.bitcast_convert"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %broadcast_in_dim = "stablehlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = array<i64: 0, 1, 2>} : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %cbrt = "stablehlo.cbrt"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %ceil = "stablehlo.ceil"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %cholesky = "stablehlo.cholesky"(%arg0) { lower = true } : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %collective_permute = "stablehlo.collective_permute"(%arg0) { source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>, channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>} : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %compare = "stablehlo.compare"(%arg0, %arg1) { comparison_direction = #stablehlo<comparison_direction LT>, compare_type = #stablehlo<comparison_type FLOAT> } : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2xi1>
  %concatenate = "stablehlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<2x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %cosine = "stablehlo.cosine"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %custom_call = "stablehlo.custom_call" (%arg0) {call_target_name = "foo"} : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %divide = "stablehlo.divide"(%arg0, %arg1) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %exponential = "stablehlo.exponential"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %exponential_minus_one = "stablehlo.exponential_minus_one"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %floor = "stablehlo.floor"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %is_finite = "stablehlo.is_finite"(%arg0) {} : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2xi1>
  %log = "stablehlo.log"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %log_plus_one = "stablehlo.log_plus_one"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %logistic = "stablehlo.logistic"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %maximum = "stablehlo.maximum"(%arg0, %arg1) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %minimum = "stablehlo.minimum"(%arg0, %arg1) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %multiply = "stablehlo.multiply"(%arg0, %arg1) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %negate = "stablehlo.negate"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %outfeed = "stablehlo.outfeed"(%arg0, %token0) {outfeed_config = ""} : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, !stablehlo.token) -> !stablehlo.token
  %optimization_barrier = "stablehlo.optimization_barrier"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>)
  %power = "stablehlo.power"(%arg0, %arg1) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %remainder = "stablehlo.remainder"(%arg0, %arg1) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %reshape = "stablehlo.reshape" (%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %rsqrt = "stablehlo.rsqrt"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %send = "stablehlo.send"(%arg0, %token0) {channel_handle = #stablehlo.channel_handle<handle = 5, type = 2>, is_host_transfer = true} : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, !stablehlo.token) -> !stablehlo.token
  %sign = "stablehlo.sign"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %sine = "stablehlo.sine"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %sqrt = "stablehlo.sqrt"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %subtract = "stablehlo.subtract"(%arg0, %arg1) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %tanh = "stablehlo.tanh"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %transpose = "stablehlo.transpose"(%arg0) {permutation = array<i64: 0, 2, 1>}: (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8<-128:127>:f32, 1.0:17>>
  %tuple = "stablehlo.tuple"(%arg0, %arg1) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tuple<tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>>
  %uniform_dequantize = "stablehlo.uniform_dequantize" (%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2xf32>
  %uniform_quantize = "stablehlo.uniform_quantize" (%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  func.return
}

func.func @batch_norm_grad_per_tensor_quantization(%input: tensor<2x2x2x2x!quant.uniform<i8:f32, 1.0:17>>, %scale: tensor<2x!quant.uniform<i8:f32, 1.0:17>>, %mean: tensor<2x!quant.uniform<i8:f32, 1.0:17>>, %variance: tensor<2x!quant.uniform<i8:f32, 1.0:17>>, %grad_output: tensor<2x2x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<2x2x2x2x!quant.uniform<i8:f32, 1.0:17>> {
  %0:3 = "stablehlo.batch_norm_grad" (%input, %scale, %mean, %variance, %grad_output)
   {epsilon = 0.001 : f32, feature_index = 0 : i64} : (tensor<2x2x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<2x!quant.uniform<i8:f32, 1.0:17>>, tensor<2x!quant.uniform<i8:f32, 1.0:17>>, tensor<2x!quant.uniform<i8:f32, 1.0:17>>, tensor<2x2x2x2x!quant.uniform<i8:f32, 1.0:17>>)
   -> (tensor<2x2x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<2x!quant.uniform<i8:f32, 1.0:17>>, tensor<2x!quant.uniform<i8:f32, 1.0:17>>)
  func.return %0#0 : tensor<2x2x2x2x!quant.uniform<i8:f32, 1.0:17>>
}


func.func @batch_norm_inference_per_tensor_quantization(%input: tensor<4x256x!quant.uniform<i8:f32, 1.0:17>>, %scale: tensor<256x!quant.uniform<i8:f32, 1.0:17>>, %offset: tensor<256x!quant.uniform<i8:f32, 1.0:17>>, %mean: tensor<256x!quant.uniform<i8:f32, 1.0:17>>, %variance: tensor<256x!quant.uniform<i8:f32, 1.0:17>>) -> (tensor<4x256x!quant.uniform<i8:f32, 1.0:17>>) {
  %0 = "stablehlo.batch_norm_inference" (%input, %scale, %offset, %mean, %variance) {
    epsilon = 1.001000e-05 : f32,
    feature_index = 1 : i64
  } : (tensor<4x256x!quant.uniform<i8:f32, 1.0:17>>, tensor<256x!quant.uniform<i8:f32, 1.0:17>>, tensor<256x!quant.uniform<i8:f32, 1.0:17>>, tensor<256x!quant.uniform<i8:f32, 1.0:17>>, tensor<256x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<4x256x!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<4x256x!quant.uniform<i8:f32, 1.0:17>>
}

func.func @batch_norm_training_per_tensor_quantization(%input: tensor<2x2x2x2x!quant.uniform<i8:f32, 1.0:17>>, %scale: tensor<2x!quant.uniform<i8:f32, 1.0:17>>, %offset: tensor<2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<2x2x2x2x!quant.uniform<i8:f32, 1.0:17>> {
  %0:3 = "stablehlo.batch_norm_training" (%input, %scale, %offset) {
    epsilon = 0.001 : f32,
    feature_index = 1 : i64
  } : (tensor<2x2x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<2x!quant.uniform<i8:f32, 1.0:17>>, tensor<2x!quant.uniform<i8:f32, 1.0:17>>) ->
      (tensor<2x2x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<2x!quant.uniform<i8:f32, 1.0:17>>, tensor<2x!quant.uniform<i8:f32, 1.0:17>>)
  func.return %0#0 : tensor<2x2x2x2x!quant.uniform<i8:f32, 1.0:17>>
}

func.func @dot_general_per_tensor_quantization(%arg0: tensor<2x3x4x!quant.uniform<i8:f32, 1.0:17>>, %arg1: tensor<2x3x5x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<2x4x5x!quant.uniform<i8:f32, 1.0:17>> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<2x3x4x!quant.uniform<i8:f32, 1.0:17>>, tensor<2x3x5x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<2x4x5x!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<2x4x5x!quant.uniform<i8:f32, 1.0:17>>
}

func.func @dynamic_slice_per_tensor_quantization(%arg0: tensor<3x4x!quant.uniform<i8:f32, 1.0:17>>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<1x4x!quant.uniform<i8:f32, 1.0:17>> {
  %0 = "stablehlo.dynamic_slice"(%arg0, %arg1, %arg2) {slice_sizes = array<i64: 1, 4>} : (tensor<3x4x!quant.uniform<i8:f32, 1.0:17>>, tensor<i64>, tensor<i64>) -> tensor<1x4x!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<1x4x!quant.uniform<i8:f32, 1.0:17>>
}

func.func @dynamic_update_slice_per_tensor_quantization(%operand: tensor<3x4x!quant.uniform<i8:f32, 1.0:17>>, %update: tensor<1x4x!quant.uniform<i8:f32, 1.0:17>>, %start_indices0: tensor<i64>, %start_indices1: tensor<i64>) -> tensor<3x4x!quant.uniform<i8:f32, 1.0:17>> {
  %0 = "stablehlo.dynamic_update_slice"(%operand, %update, %start_indices0, %start_indices1) : (tensor<3x4x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x4x!quant.uniform<i8:f32, 1.0:17>>, tensor<i64>, tensor<i64>) -> tensor<3x4x!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<3x4x!quant.uniform<i8:f32, 1.0:17>>
}

func.func @gather_per_tensor_quantization(%operand : tensor<?x?x?x?x?x?x?x?x!quant.uniform<i8:f32, 1.0:17>>, %start_indices : tensor<1x5x2xi32>) -> tensor<8x?x7x1x6x1x?x!quant.uniform<i8:f32, 1.0:17>> {
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [0, 2, 3, 4, 5],
      collapsed_slice_dims = [0, 1, 3],
      start_index_map = [0, 1],
      index_vector_dim = 2
    >,
    slice_sizes = array<i64: 1, 1, 8, 1, 7, 1, 6, 1>,
    indices_are_sorted = false
  } : (tensor<?x?x?x?x?x?x?x?x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x5x2xi32>) -> tensor<8x?x7x1x6x1x?x!quant.uniform<i8:f32, 1.0:17>>
  func.return %res : tensor<8x?x7x1x6x1x?x!quant.uniform<i8:f32, 1.0:17>>
}

func.func @map_per_tensor_quantization(%arg0: tensor<4x!quant.uniform<i8:f32, 1.0:17>>, %arg1: tensor<4x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<4x!quant.uniform<i8:f32, 1.0:17>> {
  %0 = "stablehlo.map"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<!quant.uniform<i8:f32, 1.0:17>>, %arg3: tensor<!quant.uniform<i8:f32, 1.0:17>>):
    "stablehlo.return"(%arg2) : (tensor<!quant.uniform<i8:f32, 1.0:17>>) -> ()
  }) {dimensions = array<i64: 0>} : (tensor<4x!quant.uniform<i8:f32, 1.0:17>>, tensor<4x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<4x!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<4x!quant.uniform<i8:f32, 1.0:17>>
}

func.func @pad_per_tensor_quantization(%arg0: tensor<1x2x3x!quant.uniform<i8:f32, 1.0:17>>, %arg1: tensor<!quant.uniform<i8:f32, 1.0:17>>) -> tensor<2x4x7x!quant.uniform<i8:f32, 1.0:17>> {
  %0 = "stablehlo.pad"(%arg0, %arg1) {
    edge_padding_low = array<i64: 0, 1, 2>,
    edge_padding_high = array<i64: 1, 1, 0>,
    interior_padding = array<i64: 0, 0, 1>
  } : (tensor<1x2x3x!quant.uniform<i8:f32, 1.0:17>>, tensor<!quant.uniform<i8:f32, 1.0:17>>) -> tensor<2x4x7x!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<2x4x7x!quant.uniform<i8:f32, 1.0:17>>
}

func.func @reduce_per_tensor_quantization(%arg0: tensor<16x!quant.uniform<i8:f32, 1.0:17>>, %arg1: tensor<!quant.uniform<i8:f32, 1.0:17>>) -> tensor<!quant.uniform<i8:f32, 1.0:17>> {
  %0 = "stablehlo.reduce"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<!quant.uniform<i8:f32, 1.0:17>>, %arg3: tensor<!quant.uniform<i8:f32, 1.0:17>>):
      %1 = "stablehlo.add"(%arg2, %arg3) : (tensor<!quant.uniform<i8:f32, 1.0:17>>, tensor<!quant.uniform<i8:f32, 1.0:17>>) -> tensor<!quant.uniform<i8:f32, 1.0:17>>
      "stablehlo.return"(%1) : (tensor<!quant.uniform<i8:f32, 1.0:17>>) -> ()
  }) {
    dimensions = array<i64: 0>
  } : (tensor<16x!quant.uniform<i8:f32, 1.0:17>>, tensor<!quant.uniform<i8:f32, 1.0:17>>) -> tensor<!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<!quant.uniform<i8:f32, 1.0:17>>
}

func.func @reduce_per_tensor_precision_quantization(%arg0: tensor<6x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<6x!quant.uniform<i8:f32, 1.0:17>> {
  %output = "stablehlo.reduce_precision"(%arg0) {
    exponent_bits = 5 : i32,
    mantissa_bits = 10 : i32
  } : (tensor<6x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<6x!quant.uniform<i8:f32, 1.0:17>>
  func.return %output : tensor<6x!quant.uniform<i8:f32, 1.0:17>>
}


func.func @reduce_scatter_per_tensor_quantization(%data: tensor<4x16x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<4x4x!quant.uniform<i8:f32, 1.0:17>> {
  %0 = "stablehlo.reduce_scatter"(%data) ({
    ^bb0(%arg2: tensor<!quant.uniform<i8:f32, 1.0:17>>, %arg3: tensor<!quant.uniform<i8:f32, 1.0:17>>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<!quant.uniform<i8:f32, 1.0:17>>
    "stablehlo.return"(%1) : (tensor<!quant.uniform<i8:f32, 1.0:17>>) -> ()
  }) {replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
      scatter_dimension = 1 : i64,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
      use_global_device_ids} : (tensor<4x16x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<4x4x!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<4x4x!quant.uniform<i8:f32, 1.0:17>>
}


func.func @op_reduce_window_per_tensor_quantization(%arg0: tensor<2x17x31x7x!quant.uniform<i8:f32, 0.1:-30>>, %arg1: tensor<!quant.uniform<i8:f32, 0.1:-30>>) -> tensor<2x9x16x7x!quant.uniform<i8:f32, 0.1:-30>> {
  %0 = "stablehlo.reduce_window"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<!quant.uniform<i8:f32, 0.1:-30>>, %arg3: tensor<!quant.uniform<i8:f32, 0.1:-30>>):
      %1 = "stablehlo.maximum"(%arg2, %arg3) : (tensor<!quant.uniform<i8:f32, 0.1:-30>>, tensor<!quant.uniform<i8:f32, 0.1:-30>>) -> tensor<!quant.uniform<i8:f32, 0.1:-30>>
      "stablehlo.return"(%1) : (tensor<!quant.uniform<i8:f32, 0.1:-30>>) -> ()
  }) {
    window_dimensions = array<i64: 1, 2, 2, 1>,
    window_strides = array<i64: 1, 4, 4, 1>,
    base_dilations = array<i64: 1, 2, 2, 1>,
    window_dilations = array<i64: 1, 2, 2, 1>,
    padding = dense<[[0, 0], [2, 0], [0, 2], [0, 0]]> : tensor<4x2xi64>
  } : (tensor<2x17x31x7x!quant.uniform<i8:f32, 0.1:-30>>, tensor<!quant.uniform<i8:f32, 0.1:-30>>) -> tensor<2x9x16x7x!quant.uniform<i8:f32, 0.1:-30>>
  func.return %0 : tensor<2x9x16x7x!quant.uniform<i8:f32, 0.1:-30>>
}

func.func @reverse_per_tensor_quantization(%operand: tensor<3x2x!quant.uniform<i8:f32, 0.1:-30>>) -> tensor<3x2x!quant.uniform<i8:f32, 0.1:-30>> {
  %result = "stablehlo.reverse"(%operand) {
    dimensions = array<i64: 1>
  } : (tensor<3x2x!quant.uniform<i8:f32, 0.1:-30>>) -> tensor<3x2x!quant.uniform<i8:f32, 0.1:-30>>
  func.return %result : tensor<3x2x!quant.uniform<i8:f32, 0.1:-30>>
}

func.func @round_afz_per_tensor_quantization(%arg0: tensor<2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<2x!quant.uniform<i8:f32, 1.0:17>> {
  %0 = "stablehlo.round_nearest_afz"(%arg0) {} : (tensor<2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<2x!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<2x!quant.uniform<i8:f32, 1.0:17>>
}

func.func @round_even_per_tensor_quantization(%arg0: tensor<2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<2x!quant.uniform<i8:f32, 1.0:17>> {
  %0 = "stablehlo.round_nearest_even"(%arg0) {} : (tensor<2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<2x!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<2x!quant.uniform<i8:f32, 1.0:17>>
}


func.func @scatter_per_tensor_quantization(%arg0: tensor<200x100x300x!quant.uniform<i8:f32, 1.0:17>>, %arg1: tensor<10x2xi32>, %arg2: tensor<10x300x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<200x100x300x!quant.uniform<i8:f32, 1.0:17>> {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<!quant.uniform<i8:f32, 1.0:17>>, %arg4: tensor<!quant.uniform<i8:f32, 1.0:17>>):
      %1 = "stablehlo.add"(%arg3, %arg4) : (tensor<!quant.uniform<i8:f32, 1.0:17>>, tensor<!quant.uniform<i8:f32, 1.0:17>>) -> tensor<!quant.uniform<i8:f32, 1.0:17>>
      "stablehlo.return"(%1) : (tensor<!quant.uniform<i8:f32, 1.0:17>>) -> ()
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1
    >
  } : (tensor<200x100x300x!quant.uniform<i8:f32, 1.0:17>>, tensor<10x2xi32>, tensor<10x300x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<200x100x300x!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<200x100x300x!quant.uniform<i8:f32, 1.0:17>>
}

func.func @select_per_tensor_quantization(%arg0: tensor<2x3xi1>, %arg1: tensor<2x3x!quant.uniform<i8:f32, 1.0:17>>, %arg2: tensor<2x3x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<2x3x!quant.uniform<i8:f32, 1.0:17>> {
  %0 = "stablehlo.select"(%arg0, %arg1, %arg2) : (tensor<2x3xi1>, tensor<2x3x!quant.uniform<i8:f32, 1.0:17>>, tensor<2x3x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<2x3x!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<2x3x!quant.uniform<i8:f32, 1.0:17>>
}

func.func @select_and_scatter_per_tensor_quantization(%arg0: tensor<10x24x24x64x!quant.uniform<i8:f32, 1.0:17>>, %arg1: tensor<10x23x23x64x!quant.uniform<i8:f32, 1.0:17>>, %arg2: tensor<!quant.uniform<i8:f32, 1.0:17>>) -> tensor<10x24x24x64x!quant.uniform<i8:f32, 1.0:17>> {
  %0 = "stablehlo.select_and_scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<!quant.uniform<i8:f32, 1.0:17>>, %arg4: tensor<!quant.uniform<i8:f32, 1.0:17>>):
      %1 = "stablehlo.compare"(%arg3, %arg4) {compare_type = #stablehlo<comparison_type TOTALORDER>, comparison_direction = #stablehlo<comparison_direction GE>} : (tensor<!quant.uniform<i8:f32, 1.0:17>>, tensor<!quant.uniform<i8:f32, 1.0:17>>) -> tensor<i1>
      "stablehlo.return"(%1) : (tensor<i1>) -> ()
  }, {
    ^bb0(%arg3: tensor<!quant.uniform<i8:f32, 1.0:17>>, %arg4: tensor<!quant.uniform<i8:f32, 1.0:17>>):
      %1 = "stablehlo.add"(%arg3, %arg4) : (tensor<!quant.uniform<i8:f32, 1.0:17>>, tensor<!quant.uniform<i8:f32, 1.0:17>>) -> tensor<!quant.uniform<i8:f32, 1.0:17>>
      "stablehlo.return"(%1) : (tensor<!quant.uniform<i8:f32, 1.0:17>>) -> ()
  }) {
    window_dimensions = array<i64: 1, 2, 2, 1>
  } : (tensor<10x24x24x64x!quant.uniform<i8:f32, 1.0:17>>, tensor<10x23x23x64x!quant.uniform<i8:f32, 1.0:17>>, tensor<!quant.uniform<i8:f32, 1.0:17>>) -> tensor<10x24x24x64x!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<10x24x24x64x!quant.uniform<i8:f32, 1.0:17>>
}

func.func @slice_per_tensor_qunatization(%arg0: tensor<3x4x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x!quant.uniform<i8:f32, 1.0:17>> {
  %0 = "stablehlo.slice"(%arg0) {start_indices = array<i64: 1, 0>, limit_indices = array<i64: 2, 4>, strides = array<i64: 1, 2>} : (tensor<3x4x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<1x2x!quant.uniform<i8:f32, 1.0:17>>
}

func.func @sort_per_tensor_quantization(%input0: tensor<16x16x!quant.uniform<i8:f32, 1.0:17>>, %input1: tensor<16x16x!quant.uniform<i8:f32, 1.0:17>>) {
  %0:2 = "stablehlo.sort"(%input0, %input1) ({
  ^bb0(%arg0: tensor<!quant.uniform<i8:f32, 1.0:17>>, %arg1: tensor<!quant.uniform<i8:f32, 1.0:17>>, %arg2: tensor<!quant.uniform<i8:f32, 1.0:17>>, %arg3: tensor<!quant.uniform<i8:f32, 1.0:17>>):
    %7 = "stablehlo.compare"(%arg0, %arg1) {comparison_direction = #stablehlo<comparison_direction GT>} : (tensor<!quant.uniform<i8:f32, 1.0:17>>, tensor<!quant.uniform<i8:f32, 1.0:17>>) -> tensor<i1>
    "stablehlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<16x16x!quant.uniform<i8:f32, 1.0:17>>, tensor<16x16x!quant.uniform<i8:f32, 1.0:17>>) -> (tensor<16x16x!quant.uniform<i8:f32, 1.0:17>>, tensor<16x16x!quant.uniform<i8:f32, 1.0:17>>)
  func.return
}

func.func @while_per_tensor_quantization(%arg0: tensor<4x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<?x!quant.uniform<i8:f32, 1.0:17>> {
  %while = "stablehlo.while"(%arg0) ({
  ^bb0(%arg1: tensor<?x!quant.uniform<i8:f32, 1.0:17>>):
    %1 = stablehlo.constant dense<true> : tensor<i1>
    stablehlo.return %1 : tensor<i1>
  },  {
  ^bb0(%arg1: tensor<?x!quant.uniform<i8:f32, 1.0:17>>):
    stablehlo.return %arg1 : tensor<?x!quant.uniform<i8:f32, 1.0:17>>
  }) : (tensor<4x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<?x!quant.uniform<i8:f32, 1.0:17>>
  func.return %while : tensor<?x!quant.uniform<i8:f32, 1.0:17>>
}


// -----
// Negative Tests for StableHLO OPs supporting only per-tensor quantization and not per-axis quantization


func.func @negative_abs_quantization(%arg0: tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>){
  // expected-error@+1 {{operand #0 must be ranked tensor of 4/8/16/32/64-bit signless integer or f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or complex type with 32-bit float or 64-bit float elements or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<1x2x2x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %abs_neg = "stablehlo.abs"(%arg0) : (tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:0, {0.1:-30}>>) -> tensor<1x2x2x!quant.uniform<i8<-128:127>:f32:0, {0.1:-30}>>
  func.return
}

// -----

func.func @negative_all_gather_quantization(%arg0: tensor<2x4x!quant.uniform<i8:f32:0, {0.1:-30}>>){
  // expected-error@+1 {{operand #0 must be ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or pred (AKA boolean or 1-bit integer) or 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<2x4x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %all_gather = "stablehlo.all_gather"(%arg0) { all_gather_dim = 1 : i64, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<2x4x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<2x4x!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return
}

// -----

func.func @negative_all_to_all_quantization(%arg0: tensor<2x4x!quant.uniform<i8:f32:0, {0.1:-30}>>){
  // expected-error@+1 {{operand #0 must be ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or pred (AKA boolean or 1-bit integer) or 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<2x4x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %all_to_all = "stablehlo.all_to_all"(%arg0) { split_dimension = 1 : i64, concat_dimension = 1 : i64, split_count = 2 : i64, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>} : (tensor<2x4x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<2x4x!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return
}

// -----

func.func @negative_atan_quantization(%arg0: tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>, %arg1: tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>){
  // expected-error@+1 {{operand #0 must be ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or complex type with 32-bit float or 64-bit float elements or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<1x2x2x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %atan2 = "stablehlo.atan2"(%arg0, %arg1) : (tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>, tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return
}

// -----

func.func @negative_bitcast_quantization(%arg0: tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>){
  // expected-error@+1 {{operand #0 must be ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or complex type with 32-bit float or 64-bit float elements or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<1x2x2x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %cbrt = "stablehlo.cbrt"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return
}

// -----

func.func @negative_bitcast_quantization(%arg0: tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>){
  // expected-error@+1 {{operand #0 must be ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<1x2x2x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %ceil = "stablehlo.ceil"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return
}

// -----

func.func @negative_bitcast_quantization(%arg0: tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>){
  // expected-error@+1 {{operand #0 must be ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or complex type with 32-bit float or 64-bit float elements or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<1x2x2x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %cholesky = "stablehlo.cholesky"(%arg0) { lower = true } : (tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return
}


// -----

func.func @negative_quantized_clamp(%arg0: tensor<1x!quant.uniform<ui8:f32:0, {0.1:-30}>>) -> tensor<1x!quant.uniform<ui8:f32:0, {0.1:-30}>> {
  // expected-error@+1 {{operand #0 must be ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or pred (AKA boolean or 1-bit integer) or 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<1x!quant.uniform<u8:f32:0, {1.000000e-01:-30}>>'}}
  %0 = "stablehlo.clamp"(%arg0, %arg0, %arg0) : (tensor<1x!quant.uniform<ui8:f32:0, {0.1:-30}>>, tensor<1x!quant.uniform<ui8:f32:0, {0.1:-30}>>, tensor<1x!quant.uniform<ui8:f32:0, {0.1:-30}>>) -> tensor<1x!quant.uniform<ui8:f32:0, {0.1:-30}>>
  func.return %0: tensor<1x!quant.uniform<ui8:f32:0, {0.1:-30}>>
}

// -----

func.func @negative_collective_permute_quantization(%arg0: tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>){
  // expected-error@+1 {{operand #0 must be ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or pred (AKA boolean or 1-bit integer) or 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<1x2x2x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %collective_permute = "stablehlo.collective_permute"(%arg0) { source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>, channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>} : (tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return
}

// -----

func.func @negative_compare_quantization(%arg0: tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>, %arg1: tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>){
  // expected-error@+1 {{operand #0 must be ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or pred (AKA boolean or 1-bit integer) or 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<1x2x2x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %compare = "stablehlo.compare"(%arg0, %arg1) { comparison_direction = #stablehlo<comparison_direction LT>, compare_type = #stablehlo<comparison_type FLOAT> } : (tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>, tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<1x2x2xi1>
  func.return
}

// -----

func.func @negative_concatenate_quantization(%arg0: tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>, %arg1: tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>){
  // expected-error@+1 {{operand #0 must be variadic of ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or pred (AKA boolean or 1-bit integer) or 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<1x2x2x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %concatenate = "stablehlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>, tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<2x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return
}

// -----

func.func @negative_cosine_quantization(%arg0: tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>){
  // expected-error@+1 {{operand #0 must be ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or complex type with 32-bit float or 64-bit float elements or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<1x2x2x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %cosine = "stablehlo.cosine"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return
}

// -----

func.func @negative_divide_quantization(%arg0: tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>, %arg1: tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>){
  // expected-error@+1 {{operand #0 must be ranked tensor of 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer or f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or complex type with 32-bit float or 64-bit float elements or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<1x2x2x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %divide = "stablehlo.divide"(%arg0, %arg1) : (tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>, tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return
}

// -----

func.func @negative_dynamic_slice_quantization(%arg0: tensor<3x4x!quant.uniform<i8:f32:0, {0.1:-30}>>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<1x4x!quant.uniform<i8:f32:0, {0.1:-30}>> {
  // expected-error@+1 {{operand #0 must be ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or pred (AKA boolean or 1-bit integer) or 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<3x4x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %0 = "stablehlo.dynamic_slice"(%arg0, %arg1, %arg2) {slice_sizes = array<i64: 1, 4>} : (tensor<3x4x!quant.uniform<i8:f32:0, {0.1:-30}>>, tensor<i64>, tensor<i64>) -> tensor<1x4x!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return %0 : tensor<1x4x!quant.uniform<i8:f32:0, {0.1:-30}>>
}

// -----

func.func @negative_exponential_minus_one_quantization(%arg0: tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>){
  // expected-error@+1 {{operand #0 must be ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or complex type with 32-bit float or 64-bit float elements or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<1x2x2x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %exponential_minus_one = "stablehlo.exponential_minus_one"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return
}

// -----

func.func @negative_exponential_quantization(%arg0: tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>){
  // expected-error@+1 {{operand #0 must be ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or complex type with 32-bit float or 64-bit float elements or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<1x2x2x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %exponential_minus_one = "stablehlo.exponential"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return
}

// -----

func.func @negative_floor_quantization(%arg0: tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>){
  // expected-error@+1 {{operand #0 must be ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<1x2x2x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %floor = "stablehlo.floor"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return
}

// -----

func.func @negative_floor_quantization(%arg0: tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>){
  // expected-error@+1 {{operand #0 must be ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<1x2x2x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %is_finite = "stablehlo.is_finite"(%arg0) {} : (tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<1x2x2xi1>
  func.return
}

// -----

func.func @negative_log_plus_one_quantization(%arg0: tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>){
  // expected-error@+1 {{operand #0 must be ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or complex type with 32-bit float or 64-bit float elements or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<1x2x2x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %log_plus_one = "stablehlo.log_plus_one"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return
}

// -----

func.func @negative_logistic_quantization(%arg0: tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>){
  // expected-error@+1 {{operand #0 must be ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or complex type with 32-bit float or 64-bit float elements or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<1x2x2x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %logistic = "stablehlo.logistic"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return
}

// -----

func.func @negative_log_quantization(%arg0: tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>){
  // expected-error@+1 {{operand #0 must be ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or complex type with 32-bit float or 64-bit float elements or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<1x2x2x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %log = "stablehlo.log"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return
}

// -----

func.func @negative_map_quantization(%arg0: tensor<4x!quant.uniform<i8:f32:0, {0.1:-30}>>, %arg1: tensor<4x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<4x!quant.uniform<i8:f32:0, {0.1:-30}>> {
  // expected-error@+1 {{operand #0 must be variadic of ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or pred (AKA boolean or 1-bit integer) or 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<4x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %map = "stablehlo.map"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>, %arg3: tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>):
    "stablehlo.return"(%arg2) : (tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>) -> ()
  }) {dimensions = array<i64: 0>} : (tensor<4x!quant.uniform<i8:f32:0, {0.1:-30}>>, tensor<4x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<4x!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return %map : tensor<4x!quant.uniform<i8:f32:0, {0.1:-30}>>
}

// -----

func.func @negative_maximum_quantization(%arg0: tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>, %arg1: tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>){
  // expected-error@+1 {{operand #0 must be ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or pred (AKA boolean or 1-bit integer) or 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<1x2x2x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %maximum = "stablehlo.maximum"(%arg0, %arg1) : (tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>, tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return
}

// -----

func.func @negative_minimum_quantization(%arg0: tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>, %arg1: tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>){
  // expected-error@+1 {{operand #0 must be ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or pred (AKA boolean or 1-bit integer) or 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<1x2x2x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %minimum = "stablehlo.minimum"(%arg0, %arg1) : (tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>, tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return
}

// -----

func.func @negative_multiply_quantization(%arg0: tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>, %arg1: tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>){
  // expected-error@+1 {{operand #0 must be ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or pred (AKA boolean or 1-bit integer) or 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<1x2x2x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %multiply = "stablehlo.multiply"(%arg0, %arg1) : (tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>, tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return
}

// -----

func.func @negative_negate_quantization(%arg0: tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>){
  // expected-error@+1 {{operand #0 must be ranked tensor of 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer or f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or complex type with 32-bit float or 64-bit float elements or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<1x2x2x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %negate = "stablehlo.negate"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return
}

// -----

func.func @negative_optimization_barrier_quantization(%arg0: tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>){
  // expected-error@+1 {{operand #0 must be variadic of ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or pred (AKA boolean or 1-bit integer) or 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values or token, but got 'tensor<1x2x2x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %optimization_barrier = "stablehlo.optimization_barrier"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return
}

// -----
func.func @negative_pad_quantization(%arg0: tensor<1x2x3x!quant.uniform<i8:f32:0, {0.1:-30}>>, %arg1: tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<2x4x7x!quant.uniform<i8:f32:0, {0.1:-30}>> {
  // expected-error@+1 {{operand #0 must be ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or pred (AKA boolean or 1-bit integer) or 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<1x2x3x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %pad = "stablehlo.pad"(%arg0, %arg1) {
    edge_padding_low = array<i64: 0, 1, 2>,
    edge_padding_high = array<i64: 1, 1, 0>,
    interior_padding = array<i64: 0, 0, 1>
  } : (tensor<1x2x3x!quant.uniform<i8:f32:0, {0.1:-30}>>, tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<2x4x7x!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return %pad : tensor<2x4x7x!quant.uniform<i8:f32:0, {0.1:-30}>>
}

// -----

func.func @negative_power_quantization(%arg0: tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>, %arg1: tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>){
  // expected-error@+1 {{operand #0 must be ranked tensor of 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer or f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or complex type with 32-bit float or 64-bit float elements or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<1x2x2x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %power = "stablehlo.power"(%arg0, %arg1) : (tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>, tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return
}

// -----

func.func @reduce_quantization(%arg0: tensor<16x!quant.uniform<i8:f32:0, {0.1:-30}>>, %arg1: tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<!quant.uniform<i8:f32:0, {0.1:-30}>> {
  // expected-error@+1 {{operand #0 must be variadic of ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or pred (AKA boolean or 1-bit integer) or 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<16x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %reduce = "stablehlo.reduce"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>, %arg3: tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>):
      %1 = "stablehlo.add"(%arg2, %arg3) : (tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>, tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>
      "stablehlo.return"(%1) : (tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>) -> ()
  }) {
    dimensions = array<i64: 0>
  } : (tensor<16x!quant.uniform<i8:f32:0, {0.1:-30}>>, tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return %reduce : tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>
}

// -----

func.func @negative_remainder_quantization(%arg0: tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>, %arg1: tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>){
  // expected-error@+1 {{operand #0 must be ranked tensor of 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer or f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or complex type with 32-bit float or 64-bit float elements or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<1x2x2x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %remainder = "stablehlo.remainder"(%arg0, %arg1) : (tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>, tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return
}

// -----

func.func @negative_rsqrt_quantization(%arg0: tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>){
  // expected-error@+1 {{operand #0 must be ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or complex type with 32-bit float or 64-bit float elements or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<1x2x2x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %rsqrt = "stablehlo.rsqrt"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return
}

// -----

func.func @negative_sine_quantization(%arg0: tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>){
  // expected-error@+1 {{operand #0 must be ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or complex type with 32-bit float or 64-bit float elements or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<1x2x2x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %sine = "stablehlo.sine"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return
}

// -----

func.func @negative_sqrt_quantization(%arg0: tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>){
  // expected-error@+1 {{operand #0 must be ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or complex type with 32-bit float or 64-bit float elements or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<1x2x2x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %sqrt = "stablehlo.sqrt"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return
}

// -----

func.func @negative_subtract_quantization(%arg0: tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>, %arg1: tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>){
  // expected-error@+1 {{operand #0 must be ranked tensor of 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer or f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or complex type with 32-bit float or 64-bit float elements or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<1x2x2x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %subtract = "stablehlo.subtract"(%arg0, %arg1) : (tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>, tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return
}

// -----

func.func @negative_tanh_quantization(%arg0: tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>){
  // expected-error@+1 {{operand #0 must be ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or complex type with 32-bit float or 64-bit float elements or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<1x2x2x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %tanh = "stablehlo.tanh"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<1x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return
}

// -----

func.func @negative_batch_norm_grad_quantization(%input: tensor<2x2x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>, %scale: tensor<2x!quant.uniform<i8:f32:0, {0.1:-30}>>, %mean: tensor<2x!quant.uniform<i8:f32:0, {0.1:-30}>>, %variance: tensor<2x!quant.uniform<i8:f32:0, {0.1:-30}>>, %grad_output: tensor<2x2x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<2x2x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>> {
  // expected-error@+1 {{operand #0 must be ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<2x2x2x2x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %0:3 = "stablehlo.batch_norm_grad" (%input, %scale, %mean, %variance, %grad_output)
   {epsilon = 0.001 : f32, feature_index = 0 : i64} : (tensor<2x2x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>, tensor<2x!quant.uniform<i8:f32:0, {0.1:-30}>>, tensor<2x!quant.uniform<i8:f32:0, {0.1:-30}>>, tensor<2x!quant.uniform<i8:f32:0, {0.1:-30}>>, tensor<2x2x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>)
   -> (tensor<2x2x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>, tensor<2x!quant.uniform<i8:f32:0, {0.1:-30}>>, tensor<2x!quant.uniform<i8:f32:0, {0.1:-30}>>)
  func.return %0#0 : tensor<2x2x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>
}

// -----

func.func @negative_batch_norm_inference_quantization(%input: tensor<4x256x!quant.uniform<i8:f32:0, {0.1:-30}>>, %scale: tensor<256x!quant.uniform<i8:f32, 1.0:17>>, %offset: tensor<256x!quant.uniform<i8:f32, 1.0:17>>, %mean: tensor<256x!quant.uniform<i8:f32, 1.0:17>>, %variance: tensor<256x!quant.uniform<i8:f32, 1.0:17>>) -> (tensor<4x256x!quant.uniform<i8:f32:0, {0.1:-30}>>) {
  // expected-error@+1 {{operand #0 must be ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<4x256x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %0 = "stablehlo.batch_norm_inference" (%input, %scale, %offset, %mean, %variance) {
    epsilon = 1.001000e-05 : f32,
    feature_index = 1 : i64
  } : (tensor<4x256x!quant.uniform<i8:f32:0, {0.1:-30}>>, tensor<256x!quant.uniform<i8:f32, 1.0:17>>, tensor<256x!quant.uniform<i8:f32, 1.0:17>>, tensor<256x!quant.uniform<i8:f32, 1.0:17>>, tensor<256x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<4x256x!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return %0 : tensor<4x256x!quant.uniform<i8:f32:0, {0.1:-30}>>
}
// -----

func.func @negative_batch_norm_training_quantization(%input: tensor<2x2x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>, %scale: tensor<2x!quant.uniform<i8:f32:0, {0.1:-30}>>, %offset: tensor<2x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<2x2x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>> {
  // expected-error@+1 {{operand #0 must be ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<2x2x2x2x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %0:3 = "stablehlo.batch_norm_training" (%input, %scale, %offset) {
    epsilon = 0.001 : f32,
    feature_index = 1 : i64
  } : (tensor<2x2x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>, tensor<2x!quant.uniform<i8:f32:0, {0.1:-30}>>, tensor<2x!quant.uniform<i8:f32:0, {0.1:-30}>>) ->
      (tensor<2x2x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>, tensor<2x!quant.uniform<i8:f32:0, {0.1:-30}>>, tensor<2x!quant.uniform<i8:f32:0, {0.1:-30}>>)
  func.return %0#0 : tensor<2x2x2x2x!quant.uniform<i8:f32:0, {0.1:-30}>>
}
// -----

func.func @negative_dot_general_quantization(%arg0: tensor<2x3x4x!quant.uniform<i8:f32:0, {0.1:-30}>>, %arg1: tensor<2x3x5x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<2x4x5x!quant.uniform<i8:f32:0, {0.1:-30}>> {
  // expected-error@+1 {{operand #0 must be ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or pred (AKA boolean or 1-bit integer) or 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<2x3x4x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<2x3x4x!quant.uniform<i8:f32:0, {0.1:-30}>>, tensor<2x3x5x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<2x4x5x!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return %0 : tensor<2x4x5x!quant.uniform<i8:f32:0, {0.1:-30}>>
}

// -----

func.func @negative_dynamic_update_slice_pertensor_quantization(%operand: tensor<3x4x!quant.uniform<i8:f32:0, {0.1:-30}>>, %update: tensor<1x4x!quant.uniform<i8:f32:0, {0.1:-30}>>, %start_indices0: tensor<i64>, %start_indices1: tensor<i64>) -> tensor<3x4x!quant.uniform<i8:f32, 1.0:17>> {
  // expected-error@+1 {{operand #0 must be ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or pred (AKA boolean or 1-bit integer) or 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<3x4x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %0 = "stablehlo.dynamic_update_slice"(%operand, %update, %start_indices0, %start_indices1) : (tensor<3x4x!quant.uniform<i8:f32:0, {0.1:-30}>>, tensor<1x4x!quant.uniform<i8:f32:0, {0.1:-30}>>, tensor<i64>, tensor<i64>) -> tensor<3x4x!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return %0 : tensor<3x4x!quant.uniform<i8:f32:0, {0.1:-30}>>
}

// -----

func.func @negative_gather_quantization(%operand : tensor<*x!quant.uniform<i8:f32:0, {0.1:-30}>>, %start_indices : tensor<1x5x2xi32>) -> tensor<8x?x7x1x6x1x?x!quant.uniform<i8:f32:0, {0.1:-30}>> {
  // expected-error@+1 {{operand #0 must be ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or pred (AKA boolean or 1-bit integer) or 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<*x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [0, 2, 3, 4, 5],
      collapsed_slice_dims = [0, 1, 3],
      start_index_map = [0, 1],
      index_vector_dim = 2
    >,
    slice_sizes = array<i64: 1, 1, 8, 1, 7, 1, 6, 1>,
    indices_are_sorted = false
  } : (tensor<*x!quant.uniform<i8:f32:0, {0.1:-30}>>, tensor<1x5x2xi32>) -> tensor<8x?x7x1x6x1x?x!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return %res : tensor<8x?x7x1x6x1x?x!quant.uniform<i8:f32:0, {0.1:-30}>>
}

// -----

func.func @negative_reduce_precision_quantization(%arg0: tensor<6x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<6x!quant.uniform<i8:f32:0, {0.1:-30}>> {
  // expected-error@+1 {{operand #0 must be ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<6x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %output = "stablehlo.reduce_precision"(%arg0) {
    exponent_bits = 5 : i32,
    mantissa_bits = 10 : i32
  } : (tensor<6x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<6x!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return %output : tensor<6x!quant.uniform<i8:f32:0, {0.1:-30}>>
}

// -----

func.func @negative_reduce_scatter_quantization(%data: tensor<4x16x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<4x4x!quant.uniform<i8:f32:0, {0.1:-30}>> {
  // expected-error@+1 {{operand #0 must be ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or pred (AKA boolean or 1-bit integer) or 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<4x16x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %0 = "stablehlo.reduce_scatter"(%data) ({
    ^bb0(%arg2: tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>, %arg3: tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>
    "stablehlo.return"(%1) : (tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>) -> ()
  }) {replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
      scatter_dimension = 1 : i64,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
      use_global_device_ids} : (tensor<4x16x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<4x4x!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return %0 : tensor<4x4x!quant.uniform<i8:f32:0, {0.1:-30}>>
}

// -----

func.func @negative_reduce_window_quantization(%arg0: tensor<2x17x31x7x!quant.uniform<i8:f32:0, {0.1:-30}>>, %arg1: tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<2x9x16x7x!quant.uniform<i8:f32:0, {0.1:-30}>> {
  // expected-error@+1 {{operand #0 must be variadic of ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or pred (AKA boolean or 1-bit integer) or 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<2x17x31x7x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %0 = "stablehlo.reduce_window"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>, %arg3: tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>):
      %1 = "stablehlo.maximum"(%arg2, %arg3) : (tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>, tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>
      "stablehlo.return"(%1) : (tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>) -> ()
  }) {
    window_dimensions = array<i64: 1, 2, 2, 1>,
    window_strides = array<i64: 1, 4, 4, 1>,
    base_dilations = array<i64: 1, 2, 2, 1>,
    window_dilations = array<i64: 1, 2, 2, 1>,
    padding = dense<[[0, 0], [2, 0], [0, 2], [0, 0]]> : tensor<4x2xi64>
  } : (tensor<2x17x31x7x!quant.uniform<i8:f32:0, {0.1:-30}>>, tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<2x9x16x7x!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return %0 : tensor<2x9x16x7x!quant.uniform<i8:f32:0, {0.1:-30}>>
}

// -----

func.func @negative_reverse_quantization(%operand: tensor<3x2x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<3x2x!quant.uniform<i8:f32:0, {0.1:-30}>> {
  // expected-error@+1 {{operand #0 must be ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or pred (AKA boolean or 1-bit integer) or 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<3x2x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %result = "stablehlo.reverse"(%operand) {
    dimensions = array<i64: 1>
  } : (tensor<3x2x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<3x2x!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return %result : tensor<3x2x!quant.uniform<i8:f32:0, {0.1:-30}>>
}

// -----

func.func @negative_round_afz(%arg0: tensor<2x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<2x!quant.uniform<i8:f32:0, {0.1:-30}>> {
  // expected-error@+1 {{ operand #0 must be ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<2x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %0 = "stablehlo.round_nearest_afz"(%arg0) {} : (tensor<2x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<2x!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return %0 : tensor<2x!quant.uniform<i8:f32:0, {0.1:-30}>>
}

// -----

func.func @negative_round_even(%arg0: tensor<2x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<2x!quant.uniform<i8:f32:0, {0.1:-30}>> {
  // expected-error@+1 {{ operand #0 must be ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<2x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %0 = "stablehlo.round_nearest_even"(%arg0) {} : (tensor<2x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<2x!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return %0 : tensor<2x!quant.uniform<i8:f32:0, {0.1:-30}>>
}

// -----

func.func @negative_scatter_quantization(%arg0: tensor<200x100x300x!quant.uniform<i8:f32:0, {0.1:-30}>>, %arg1: tensor<10x2xi32>, %arg2: tensor<10x300x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<200x100x300x!quant.uniform<i8:f32:0, {0.1:-30}>> {
  // expected-error@+1 {{operand #0 must be variadic of ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or pred (AKA boolean or 1-bit integer) or 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<200x100x300x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>, %arg4: tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>):
      %1 = "stablehlo.add"(%arg3, %arg4) : (tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>, tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>
      "stablehlo.return"(%1) : (tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>) -> ()
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1
    >
  } : (tensor<200x100x300x!quant.uniform<i8:f32:0, {0.1:-30}>>, tensor<10x2xi32>, tensor<10x300x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<200x100x300x!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return %0 : tensor<200x100x300x!quant.uniform<i8:f32:0, {0.1:-30}>>
}

// -----

func.func @negative_select_quantization(%arg0: tensor<2x3xi1>, %arg1: tensor<2x3x!quant.uniform<i8:f32:0, {0.1:-30}>>, %arg2: tensor<2x3x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<2x3x!quant.uniform<i8:f32:0, {0.1:-30}>> {
  // expected-error@+1 {{operand #1 must be ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or pred (AKA boolean or 1-bit integer) or 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<2x3x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %0 = "stablehlo.select"(%arg0, %arg1, %arg2) : (tensor<2x3xi1>, tensor<2x3x!quant.uniform<i8:f32:0, {0.1:-30}>>, tensor<2x3x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<2x3x!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return %0 : tensor<2x3x!quant.uniform<i8:f32:0, {0.1:-30}>>
}

// -----

func.func @negative_slice_quantization(%arg0: tensor<3x4x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<1x2x!quant.uniform<i8:f32:0, {0.1:-30}>> {
  // expected-error@+1 {{operand #0 must be ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or pred (AKA boolean or 1-bit integer) or 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<3x4x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %0 = "stablehlo.slice"(%arg0) {start_indices = array<i64: 1, 0>, limit_indices = array<i64: 2, 4>, strides = array<i64: 1, 2>} : (tensor<3x4x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<1x2x!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return %0 : tensor<1x2x!quant.uniform<i8:f32:0, {0.1:-30}>>
}

// -----


func.func @negative_sort_quantization(%input0: tensor<16x16x!quant.uniform<i8:f32:0, {0.1:-30}>>, %input1: tensor<16x16x!quant.uniform<i8:f32:0, {0.1:-30}>>) {
  // expected-error@+1 {{operand #0 must be variadic of ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or pred (AKA boolean or 1-bit integer) or 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<16x16x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %0:2 = "stablehlo.sort"(%input0, %input1) ({
  ^bb0(%arg0: tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>, %arg1: tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>, %arg2: tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>, %arg3: tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>):
    %7 = "stablehlo.compare"(%arg0, %arg1) {comparison_direction = #stablehlo<comparison_direction GT>} : (tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>, tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<i1>
    "stablehlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<16x16x!quant.uniform<i8:f32:0, {0.1:-30}>>, tensor<16x16x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> (tensor<16x16x!quant.uniform<i8:f32:0, {0.1:-30}>>, tensor<16x16x!quant.uniform<i8:f32:0, {0.1:-30}>>)
  func.return
}

// -----

func.func @negative_select_and_scatter_quantization(%arg0: tensor<10x24x24x64x!quant.uniform<i8:f32:0, {0.1:-30}>>, %arg1: tensor<10x23x23x64x!quant.uniform<i8:f32:0, {0.1:-30}>>, %arg2: tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<10x24x24x64x!quant.uniform<i8:f32:0, {0.1:-30}>> {
  // expected-error@+1 {{operand #0 must be ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or pred (AKA boolean or 1-bit integer) or 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<10x24x24x64x!quant.uniform<i8:f32:0, {1.000000e-01:-30}>>'}}
  %0 = "stablehlo.select_and_scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>, %arg4: tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>):
      %1 = "stablehlo.compare"(%arg3, %arg4) {compare_type = #stablehlo<comparison_type TOTALORDER>, comparison_direction = #stablehlo<comparison_direction GE>} : (tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>, tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<i1>
      "stablehlo.return"(%1) : (tensor<i1>) -> ()
  }, {
    ^bb0(%arg3: tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>, %arg4: tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>):
      %1 = "stablehlo.add"(%arg3, %arg4) : (tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>, tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>
      "stablehlo.return"(%1) : (tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>) -> ()
  }) {
    window_dimensions = array<i64: 1, 2, 2, 1>
  } : (tensor<10x24x24x64x!quant.uniform<i8:f32:0, {0.1:-30}>>, tensor<10x23x23x64x!quant.uniform<i8:f32:0, {0.1:-30}>>, tensor<!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<10x24x24x64x!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return %0 : tensor<10x24x24x64x!quant.uniform<i8:f32:0, {0.1:-30}>>
}

// -----

func.func @illegal_storage_type_for_quantized_element_type(%arg0: tensor<4x!quant.uniform<si8:f32, 1.000000e+00>>) -> tensor<4xf32> {
  // expected-error@+1 {{operand #0 must be tensor of 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer or 4/8/16/32-bit uniform quantized per axis signed integer or 4/8/16/32-bit uniform quantized per axis unsigned integer values, but got 'tensor<4x!quant.uniform<i8:f32, 1.000000e+00>>}}
  %0 = "stablehlo.uniform_dequantize"(%arg0) : (tensor<4x!quant.uniform<si8:f32, 1.000000e+00>>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// -----

func.func @convolution_2d(%arg0: tensor<1x8x8x207x!quant.uniform<i8:f32, 2.0:15>>, %arg1: tensor<3x3x207x16x!quant.uniform<i8:f32, 5.0:20>>) -> tensor<1x8x8x16x!quant.uniform<i8:f32, 10.0:50>> {
  %0 = stablehlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
         {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} :
       (tensor<1x8x8x207x!quant.uniform<i8:f32, 2.0:15>>, tensor<3x3x207x16x!quant.uniform<i8:f32, 5.0:20>>) -> tensor<1x8x8x16x!quant.uniform<i8:f32, 10.0:50>>
  func.return %0 : tensor<1x8x8x16x!quant.uniform<i8:f32, 10.0:50>>
}

// -----

func.func @convolution_c28(%arg0: tensor<1x8x8x207xf32>, %arg1: tensor<3x3x207x16x!quant.uniform<i8:f32, 5.0:20>>) -> tensor<1x8x8x16x!quant.uniform<i8:f32, 10.0:50>> {
  // expected-error@+1 {{not all of operands and result are quantized}}
  %0 = stablehlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
         {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} :
       (tensor<1x8x8x207xf32>, tensor<3x3x207x16x!quant.uniform<i8:f32, 5.0:20>>) -> tensor<1x8x8x16x!quant.uniform<i8:f32, 10.0:50>>
  func.return %0 : tensor<1x8x8x16x!quant.uniform<i8:f32, 10.0:50>>
}

// -----

func.func @convolution_c29(%arg0: tensor<1x8x8x207x!quant.uniform<i16:f32, 2.0:15>>, %arg1: tensor<3x3x207x16x!quant.uniform<i8:f32, 5.0:20>>) -> tensor<1x8x8x16x!quant.uniform<i8:f32, 10.0:50>> {
  // expected-error@+1 {{mismatched operand storage types 'i16' and 'i8'}}
  %0 = stablehlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
         {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} :
       (tensor<1x8x8x207x!quant.uniform<i16:f32, 2.0:15>>, tensor<3x3x207x16x!quant.uniform<i8:f32, 5.0:20>>) -> tensor<1x8x8x16x!quant.uniform<i8:f32, 10.0:50>>
  func.return %0 : tensor<1x8x8x16x!quant.uniform<i8:f32, 10.0:50>>
}

// -----

func.func @convolution_c30(%arg0: tensor<1x8x8x207x!quant.uniform<i8:f64, 2.0:15>>, %arg1: tensor<3x3x207x16x!quant.uniform<i8:f32, 5.0:20>>) -> tensor<1x8x8x16x!quant.uniform<i8:f32, 10.0:50>> {
  // expected-error@+1 {{mismatched operands and result expressed types}}
  %0 = stablehlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
         {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} :
       (tensor<1x8x8x207x!quant.uniform<i8:f64, 2.0:15>>, tensor<3x3x207x16x!quant.uniform<i8:f32, 5.0:20>>) -> tensor<1x8x8x16x!quant.uniform<i8:f32, 10.0:50>>
  func.return %0 : tensor<1x8x8x16x!quant.uniform<i8:f32, 10.0:50>>
}

// -----

func.func @convolution_c31(%arg0: tensor<1x8x8x207x!quant.uniform<i8:f32, 2.0:15>>, %arg1: tensor<3x3x207x16x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<1x8x8x16x!quant.uniform<i8:f32, 10.0:50>> {
  // expected-error@+1 {{operand and result are of mixed per_tensor and per_axis quantized tensor type}}
  %0 = stablehlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
         {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} :
       (tensor<1x8x8x207x!quant.uniform<i8:f32, 2.0:15>>, tensor<3x3x207x16x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<1x8x8x16x!quant.uniform<i8:f32, 10.0:50>>
  func.return %0 : tensor<1x8x8x16x!quant.uniform<i8:f32, 10.0:50>>
}

// -----

func.func @convolution_c32(%arg0: tensor<1x8x8x207x!quant.uniform<i8:f32, 2.0:15>>, %arg1: tensor<3x3x207x16x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<1x8x8x16x!quant.uniform<i8:f32:0, {0.1:-30}>> {
  // expected-error@+1 {{mismatched kernel_output_feature_dimension 3 and operand quantized dimension 0}}
  %0 = stablehlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
         {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} :
       (tensor<1x8x8x207x!quant.uniform<i8:f32, 2.0:15>>, tensor<3x3x207x16x!quant.uniform<i8:f32:0, {0.1:-30}>>) -> tensor<1x8x8x16x!quant.uniform<i8:f32:0, {0.1:-30}>>
  func.return %0 : tensor<1x8x8x16x!quant.uniform<i8:f32:0, {0.1:-30}>>
}

// -----

func.func @convolution_c33(%arg0: tensor<1x8x8x207x!quant.uniform<i8:f32, 2.0:15>>, %arg1: tensor<3x3x207x16x!quant.uniform<i8:f32:3, {0.1:-30}>>) -> tensor<1x8x8x16x!quant.uniform<i8:f32:0, {2.0:-30}>> {
 // expected-error@+1 {{mismatched kernel_output_feature_dimension 3 and result quantized dimension 0}}
 %0 = stablehlo.convolution(%arg0, %arg1)
     dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
     window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
     {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} :
    (tensor<1x8x8x207x!quant.uniform<i8:f32, 2.0:15>>, tensor<3x3x207x16x!quant.uniform<i8:f32:3, {0.1:-30}>>) -> tensor<1x8x8x16x!quant.uniform<i8:f32:0, {2.0:-30}>>
 func.return %0 : tensor<1x8x8x16x!quant.uniform<i8:f32:0, {2.0:-30}>>
}