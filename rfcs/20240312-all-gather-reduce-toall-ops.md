# [RFC] Standardize AllGatherOp, AllReduceOp, AllToAllOp OPs for HLO-StableHLO parity
Status: Draft<br/>
Initial version: 03/12/2024<br/>
Last updated: 03/12/2024<br/>
Discussion thread: [GitHub](add PR Link)


## Existing Specification
The OPs are categorized as StableHLO [Collective OPs](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#collective-ops). They split the processes within the StableHLO process grid into StableHLO process groups and execute a joint computation within each process group, independently from other process groups. Existing spec is limited to **single-operand** and **single-result**.

## Motivation
MHLO OPs already [support](https://github.com/tensorflow/mlir-hlo/blob/master/mhlo/IR/hlo_ops.td) **multi-operand** and **multi-result** which is in sync with xla semantics [`all_reduce`](https://openxla.org/xla/operation_semantics#allreduce) [`all-gather`](https://openxla.org/xla/operation_semantics#allgather) and [`alltoall`](https://openxla.org/xla/operation_semantics#alltoall) which supports multi-operand and multi-result.

AllReduceOp support is requested at open ticket [#1](https://github.com/openxla/stablehlo/issues/1370).
AllToAllOp support is requested at open ticket [#2](https://github.com/openxla/stablehlo/issues/574) and identified as a feature gap.

## Proposed Specification

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

## Open Questions
Q. need to elaborate on "Supporting multi-operand and multi-result for these OPs is useful for horizontal scaling" ?

Q. other [Collective OPs](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#collective-ops) also need / will need similar support?
