# [RFC] Standardize TanOp, CustomCall with typed FFI, and tuple-collectives for HLO-StableHLO parity, used by JAX and PT/XLA
Status: Draft<br/>
Initial version: 03/12/2024<br/>
Last updated: 03/12/2024<br/>
Discussion thread: [GitHub](add PR Link)

## Motivation

`TanOP`, `CustomCallOP` with typed FFI and tuple-collectives (`all_gather`, `all_reduce`, `alltoall`) OPs are already successful in MHLO. There are hacks to leverage these features (unregistered attributes, serialize strings, etc) which are not required once we standardize these OPs in StableHLO. There are existing user requests in the StableHLO repo for these features.

### TanOP

Frameworks and Compilers both want TanOP.
Jax has [`jnp.tan`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.tan.html), PyTorch has [`torch.tan`](https://pytorch.org/docs/stable/generated/torch.tan.html). On Compilers side, XLA has [`mhlo.tan`](https://github.com/tensorflow/mlir-hlo/blob/master/mhlo/IR/hlo_ops.td#L633).

Adding TanOP to StableHLO is requested at ticket [#1](https://github.com/openxla/stablehlo/issues/1358)

### CustomCallOp with typed FFI

StableHLO CustomCallOp to support `API_VERSION_TYPED_FFI` as `StableHLO_CustomCallApiVersionAttr`. It will help to unify metadata under single `mlir::DictionaryAttr` and won't need to pass the metadata as strings. Similar to what [MHLO CustomCallOp](https://github.com/tensorflow/mlir-hlo/blob/master/mhlo/IR/hlo_ops.td#L2483) is already doing. 
open tickets for this request: [#2](https://github.com/openxla/stablehlo/issues/637), [#3](https://github.com/openxla/stablehlo/issues/741)

### Tuple-collectives (AllGatherOp, AllReduceOp, AllToAllOp)

MHLO OPs already [support](https://github.com/tensorflow/mlir-hlo/blob/master/mhlo/IR/hlo_ops.td) **multi-operand** and **multi-result** which is in sync with xla semantics [`all_reduce`](https://openxla.org/xla/operation_semantics#allreduce) [`all-gather`](https://openxla.org/xla/operation_semantics#allgather) and [`alltoall`](https://openxla.org/xla/operation_semantics#alltoall) which supports multi-operand and multi-result.

AllReduceOp support is requested at open ticket [#4](https://github.com/openxla/stablehlo/issues/1370).
AllToAllOp support is requested at open ticket [#5](https://github.com/openxla/stablehlo/issues/574) and identified as a feature gap.


## Proposed Specification

### tan

#### Semantics

Performs element-wise tangent operation on `operand` tensor and
produces a `result` tensor. Depending on the element type, does the following:

* For floats: `tan` from IEEE-754.
* For complex numbers: complex tangent.
* For quantized types:
  * `dequantize_op_quantize(tan, operand, type(result))`.

#### Inputs

| Label | Name      | Type                                                                    | Constraints |
|-------|-----------|-------------------------------------------------------------------------|-------------|
| (I1)  | `operand` | tensor of floating-point or complex type or per-tensor quantized tensor | (C1)        |

#### Outputs

| Name     | Type                                                                    | Constraints |
|----------|-------------------------------------------------------------------------|-------------|
| `result` | tensor of floating-point or complex type or per-tensor quantized tensor | (C1)        |

#### Constraints

* (C1) `baseline_type(operand) = baseline_type(result)`.

#### Examples

```mlir
// %operand: [-1.0, 0.0, 1.0]
%result = "stablehlo.tan"(%operand) : (tensor<3xf32>) -> tensor<3xf32>
// %result: [-0.76159416, 0.0, 0.76159416]
```



### custom_call

#### Semantics

Encapsulates an implementation-defined operation `call_target_name` that takes
`inputs` and `called_computations` and produces `results`. `has_side_effect`,
`backend_config` and `api_version` may be used to provide additional
implementation-defined metadata.

#### Inputs

| Label | Name                  | Type                                          |
|-------|-----------------------|-----------------------------------------------|
| (I1)  | `inputs`              | variadic number of values                     |
| (I2)  | `call_target_name`    | constant of type `string`                     |
| (I3)  | `has_side_effect`     | constant of type `i1`                         |
| (I4)  | `backend_config`      | constant of type `string`                     |
| (I5)  | `api_version`         | constant of type `si32`                       |
| (I6)  | `called_computations` | variadic number of constants of type `string` |

#### Outputs

| Name      | Type                      |
|-----------|---------------------------|
| `results` | variadic number of values |

#### Examples

```mlir
%results = "stablehlo.custom_call"(%input0) {
  call_target_name = "foo",
  has_side_effect = false,
  backend_config = "bar",
  api_version = 1 : i32,
  called_computations = [@foo]
} : (tensor<f64>) -> tensor<f64>
```



### all_gather

#### Semantics

Within each process group in the StableHLO process grid, concatenates the values
of the `operands` tensors from each process along `all_gather_dim` and produces a
`results` tensors.

The operation splits the StableHLO process grid into `process_groups` which is
defined as follows:

* `cross_replica(replica_groups)`
  if `channel_id <= 0 and use_global_device_ids = false`.
* `cross_replica_and_partition(replica_groups)`
  if `channel_id > 0 and use_global_device_ids = false`.
* `flattened_ids(replica_groups)`
  if `channel_id > 0 and use_global_device_ids = true`.

Afterwards, within each `process_group`:

* `operands@receiver = [operand@sender for sender in process_group]` for all
  `receiver` in `process_group`.
* `results@process = concatenate(operands@process, all_gather_dim)` for all
  `process` in `process_group`.

#### Inputs

| Label | Name                    | Type                                         | Constraints |
|-------|-------------------------|----------------------------------------------|-------------|
| (I1)  | `operands`               | variadic tensors or per-tensor quantized tensors        | (C1), (C6)  |
| (I2)  | `all_gather_dim`        | constant of type `si64`                      | (C1), (C6)  |
| (I3)  | `replica_groups`        | 2-dimensional tensor constant of type `si64` | (C2-C4)     |
| (I4)  | `channel_id`            | constant of type `si64`                      | (C5)        |
| (I5)  | `use_global_device_ids` | constant of type `i1`                        | (C5)        |

#### Outputs

| Name     | Type                                  | Constraints |
|----------|---------------------------------------|-------------|
| `results` | variadic tensors or per-tensor quantized tensors | (C6)        |

#### Constraints

* (C1) `0 <= all_gather_dim < rank(operand)`.
* (C2) `is_unique(replica_groups)`.
* (C3) `size(replica_groups)` is defined as:
  * `num_replicas` if `cross_replica` is used.
  * `num_replicas` if `cross_replica_and_partition` is used.
  * `num_processes` if `flattened_ids` is used.
* (C4) `0 <= replica_groups < size(replica_groups)`.
* (C5) If `use_global_device_ids = true`, then `channel_id > 0`.
* (C6) `type(results...) = type(operands...)` except:
  * `dim(results..., all_gather_dim) =
    dim(operands..., all_gather_dim) * dim(process_groups, 1)`.

#### Examples

```mlir
// num_replicas: 2
// num_partitions: 1
// %operand1@(0, 0): [[1, 2], [3, 4]]
// %operand1@(1, 0): [[5, 6], [7, 8]]
%results = "stablehlo.all_gather"(%operand1) {
  all_gather_dim = 1 : i64,
  replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
  // channel_id = 0
  channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
  // use_global_device_ids = false
} : (tensor<2x2xi64>) -> tensor<2x4xi64>
// %result1@(0, 0): [[1, 2, 5, 6], [3, 4, 7, 8]]
// %result1@(1, 0): [[1, 2, 5, 6], [3, 4, 7, 8]]
```

### all_reduce


### alltoall
