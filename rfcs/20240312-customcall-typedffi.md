# [RFC] Standardize CustomCallOp for HLO-StableHLO parity
Status: Draft<br/>
Initial version: 03/12/2024<br/>
Last updated: 03/12/2024<br/>
Discussion thread: [GitHub](add PR Link)

## Motivation

StableHLO CustomCallOp to support `API_VERSION_TYPED_FFI` as `StableHLO_CustomCallApiVersionAttr`. It will help to unify metadata under single `mlir::DictionaryAttr` and won't need to pass the metadata as strings. Similar to what [MHLO CustomCallOp](https://github.com/tensorflow/mlir-hlo/blob/master/mhlo/IR/hlo_ops.td#L2483) is already doing. *A dictionary allows us to do some interesting things with compatibility, namely if the attrs in it are StableHLO attributes, they can be upgraded/downgraded by our compatibility machinery, providing stability will less reliance on compiler back-ends than a string argument.*
open tickets for this request: [#1](https://github.com/openxla/stablehlo/issues/637), [#2](https://github.com/openxla/stablehlo/issues/741)

## Proposed Specification

### custom_call

#### Semantics

Encapsulates an implementation-defined operation `call_target_name` that takes
`inputs` and `called_computations` and produces `results`. `has_side_effect`,
`backend_config` and `api_version` may be used to provide additional
implementation-defined metadata.

For `API_VERSION_TYPED_FFI` custom calls `backend_config` must be a
dictionary attribute, that will be encoded according to the custom call
calling convention and passed to the external function as the attributes
argument. External code is expected to use declarative bindings (see
`xla/runtime/custom_call.h`) to decode them at run time. These custom
calls are only supported if XLA uses XLA runtime.

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
