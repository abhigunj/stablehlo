# [RFC] Standardize TanOp, CustomCallOp with typed FFI, and tuple-collectives ops

Status: Draft<br/>
Initial version: 03/12/2024<br/>
Last updated: 03/12/2024<br/>
Discussion thread: [GitHub](add PR Link)

## Motivation

MHLO `tan` op, `custom_call` op with typed FFI and tuple-collectives ops
(`all_gather`, `all_reduce`, `all_to_all`) supports features which are
successfully being used by JAX and PT/XLA. Standardizing StableHLO ops
will ensure HLO-StableHLO feature parity. There are hacks in place to leverage
these features (unregistered attributes, serialize strings) and standardizing
the ops is a hack-free solution. Also, there are existing user requests in the
StableHLO repo for these features.

### TanOp

Frameworks and Compilers both want `tan` op.
Jax has [`jnp.tan`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.tan.html),
PyTorch has [`torch.tan`](https://pytorch.org/docs/stable/generated/torch.tan.html).
On Compilers side, XLA has [`mhlo.tan`](https://github.com/tensorflow/mlir-hlo/blob/master/mhlo/IR/hlo_ops.td#L633).
StableHLO doesn't support `tan` op. Open ticket for this request
[#1](https://github.com/openxla/stablehlo/issues/1358)

### CustomCallOp with typed FFI

StableHLO `custom_call` op to support `API_VERSION_TYPED_FFI` as an enum value [`StableHLO_CustomCallApiVersionAttr`](https://github.com/openxla/stablehlo/blob/04365f85cfbffe3d95ba2fb79ff34cd929d4a9a6/stablehlo/dialect/StablehloEnums.td#L88).
It will help to unify metadata under single `mlir::DictionaryAttr`. Same as what
[MHLO custom_call op](https://github.com/tensorflow/mlir-hlo/blob/master/mhlo/IR/hlo_ops.td#L2483)
has already enabled. Open tickets for this request: [#2](https://github.com/openxla/stablehlo/issues/637),
[#3](https://github.com/openxla/stablehlo/issues/741)

### Tuple-collectives (AllGatherOp, AllReduceOp, AllToAllOp)

StableHLO tuple-collective ops support is limited to **single-operand** and **single-result**.
MHLO ops [support](https://github.com/tensorflow/mlir-hlo/blob/master/mhlo/IR/hlo_ops.td)
**multi-operand** and **multi-result** which is in sync with xla semantics
[`all_reduce`](https://openxla.org/xla/operation_semantics#allreduce)
[`all_gather`](https://openxla.org/xla/operation_semantics#allgather) and
[`all_to_all`](https://openxla.org/xla/operation_semantics#alltoall) which
supports multi-operand and multi-result. `all_reduce` support is requested
at open ticket [#4](https://github.com/openxla/stablehlo/issues/1370).
`all_to_all` support is requested at open ticket
[#5](https://github.com/openxla/stablehlo/issues/574) and identified as a feature
gap.

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
// %result: [-1.55740772465, 0.0, 1.55740772465]
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
  backend_config = {bar = 42 : i32},
  api_version = 4 : i32,
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

Afterwards, within each process_group and for each i in [0, size(operands)]:

* `operands[i]@receiver = [operand[i]@sender for sender in process_group]` for all
  `receiver` in `process_group`.
* `results[i]@process = concatenate(operands[i]@process, all_gather_dim)` for all
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

* (C1) `0 <= all_gather_dim < rank(operands...)`.
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
// %operand@(0, 0): [[1, 2], [3, 4]]
// %operand@(1, 0): [[5, 6], [7, 8]]
%result = "stablehlo.all_gather"(%operand) {
  all_gather_dim = 1 : i64,
  replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
  // channel_id = 0
  channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
  // use_global_device_ids = false
} : (tensor<2x2xi64>) -> tensor<2x4xi64>
// %result@(0, 0): [[1, 2, 5, 6], [3, 4, 7, 8]]
// %result@(1, 0): [[1, 2, 5, 6], [3, 4, 7, 8]]
```

### all_reduce

#### Semantics

Within each process group in the StableHLO process grid, applies a reduction
function `computation` to the values of the `operands` tensors from each process
and produces a `results` tensors.

The operation splits the StableHLO process grid into `process_groups` which is
defined as follows:

* `cross_replica(replica_groups)`
  if `channel_id <= 0 and use_global_device_ids = false`.
* `cross_replica_and_partition(replica_groups)`
  if `channel_id > 0 and use_global_device_ids = false`.
* `flattened_ids(replica_groups)`
  if `channel_id > 0 and use_global_device_ids = true`.

Afterwards, within each `process_group` and for each i in [0, size(operands)]:

* `results[i]@process[results[i]_index] = exec(schedule)` for some binary tree
  `schedule` where:
  * `exec(node)` = `computation(exec(node.left), exec(node.right))`.
  * `exec(leaf)` = `leaf.value`.
* `schedule` is an implementation-defined binary tree whose in-order
  traversal is `to_destination_type(operands[i]@process_group...[results[i]_index],
  type(func_inputs(computation)[0]))`.

#### Inputs

| Label | Name                    | Type                                                             | Constraints |
|-------|-------------------------|------------------------------------------------------------------|-------------|
| (I1)  | `operands`               | variadic tensors or per-tensor quantized tensors                            | (C5), (C6)  |
| (I2)  | `replica_groups`        | variadic number of 1-dimensional tensor constants of type `si64` | (C1-C3)     |
| (I3)  | `channel_id`            | constant of type `si64`                                          | (C4)        |
| (I4)  | `use_global_device_ids` | constant of type `i1`                                            | (C4)        |
| (I5)  | `computation`           | function                                                         | (C5)        |

#### Outputs

| Name     | Type                                  | Constraints |
|----------|---------------------------------------|-------------|
| `results` | variadic tensors or per-tensor quantized tensors | (C6-C7)     |

#### Constraints

* (C1) `is_unique(replica_groups)`.
* (C2) `size(replica_groups)` is defined as:
  * `num_replicas` if `cross_replica` is used.
  * `num_replicas` if `cross_replica_and_partition` is used.
  * `num_processes` if `flattened_ids` is used.
* (C3) `0 <= replica_groups < size(replica_groups)`.
* (C4) If `use_global_device_ids = true`, then `channel_id > 0`.
* (C5) `computation` has type `(tensor<E>, tensor<E>) -> (tensor<E>)` where
       `is_promotable(element_type(operand), E)`.
* (C6) `shape(results...) = shape(operands...)`.
* (C7) `element_type(results...) = E`.

#### Examples

```mlir
// num_replicas: 2
// num_partitions: 1
// %operand@(0, 0): [1, 2, 3, 4]
// %operand@(1, 0): [5, 6, 7, 8]
%result = "stablehlo.all_reduce"(%operand) ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
    %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    "stablehlo.return"(%0) : (tensor<i64>) -> ()
}) {
  replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
  channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
} : (tensor<4xi64>) -> tensor<4xi64>
// %result@(0, 0): [6, 8, 10, 12]
// %result@(1, 0): [6, 8, 10, 12]
```

### all_to_all

#### Semantics

![](images/spec/all_to_all.svg)

Within each process group in the StableHLO process grid, splits the values of
the `operands` tensors along `split_dimension` into parts, scatters the split
parts between the processes, concatenates the scattered parts along
`concat_dimension` and produces a `results` tensors.

The operation splits the StableHLO process grid into `process_groups` which is
defined as follows:

* `cross_replica(replica_groups)` if `channel_id <= 0`.
* `cross_partition(replica_groups)` if `channel_id > 0`.

Afterwards, within each `process_group` and for each i in [0, size(operands)]:

* `split_parts@sender = split(operands[i]@sender, split_count, split_dimension)`
  for all `sender` in `process_group`.
* `scattered_parts@receiver = [split_parts@sender[receiver_index] for
  sender in process_group]` where
  `receiver_index = process_group.index(receiver)`.
* `results[i]@process = concatenate(scattered_parts@process, concat_dimension)`.

#### Inputs

| Label | Name               | Type                                         | Constraints            |
|-------|--------------------|----------------------------------------------|------------------------|
| (I1)  | `operands`          | tensors or per-tensor quantized tensors        | (C1-C3), (C9)          |
| (I2)  | `split_dimension`  | constant of type `si64`                      | (C1), (C2), (C9)       |
| (I3)  | `concat_dimension` | constant of type `si64`                      | (C3), (C9)             |
| (I4)  | `split_count`      | constant of type `si64`                      | (C2), (C4), (C8), (C9) |
| (I5)  | `replica_groups`   | 2-dimensional tensor constant of type `si64` | (C5-C8)                |
| (I6)  | `channel_id`       | constant of type `si64`                      |                        |

#### Outputs

| Name     | Type                                  | Constraints |
|----------|---------------------------------------|-------------|
| `results` | tensors or per-tensor quantized tensors | (C9)        |

#### Constraints

* (C1) `0 <= split_dimension < rank(operands...)`.
* (C2) `dim(operands..., split_dimension) % split_count = 0`.
* (C3) `0 <= concat_dimension < rank(operands...)`.
* (C4) `0 < split_count`.
* (C5) `is_unique(replica_groups)`.
* (C6) `size(replica_groups)` is defined as:
  * `num_replicas` if `cross_replica` is used.
  * `num_partitions` if `cross_partition` is used.
* (C7) `0 <= replica_groups < size(replica_groups)`.
* (C8) `dim(replica_groups, 1) = split_count`.
* (C9) `type(results...) = type(operands...)` except:
  * `dim(results..., split_dimension) =
    dim(operands..., split_dimension) / split_count`.
  * `dim(results..., concat_dimension) =
    dim(operands..., concat_dimension) * split_count`.

#### Examples

```mlir
// num_replicas: 2
// num_partitions: 1
// %operand@(0, 0): [[1, 2, 3, 4],
//                   [5, 6, 7, 8]]
// %operand@(1, 0): [[9, 10, 11, 12],
//                   [13, 14, 15, 16]]
%result = "stablehlo.all_to_all"(%operand) {
  split_dimension = 1 : i64,
  concat_dimension = 0 : i64,
  split_count = 2 : i64,
  replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
} : (tensor<2x4xi64>) -> tensor<4x2xi64>
// %result@(0, 0): [[1, 2],
//                  [5, 6],
//                  [9, 10],
//                  [13, 14]]
// %result@(1, 0): [[3, 4],
//                  [7, 8],
//                  [11, 12],
//                  [15, 16]]
```
