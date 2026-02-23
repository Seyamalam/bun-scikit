# Native ABI Contract

This document defines the stable ABI boundary between JavaScript runtimes (Bun/Node) and the Zig compute core.

## ABI Version

- Exported symbol: `bun_scikit_abi_version() -> u32`
- Current version: `1`
- JavaScript bridges must refuse to load mismatched ABI versions.

## Status Codes

Zig exports numeric status constants:

- `bun_scikit_status_ok()`
- `bun_scikit_status_invalid_handle()`
- `bun_scikit_status_invalid_shape()`
- `bun_scikit_status_allocation_failed()`
- `bun_scikit_status_fit_failed()`
- `bun_scikit_status_symbol_unavailable()`

## Handle Lifecycle

All model handles are opaque native pointers represented as `usize` in native code and `BigInt` in JS.

Lifecycle:

1. `*_model_create(...) -> handle`
2. `*_model_fit(...)` / `*_model_predict(...)` / `*_model_copy_coefficients(...)`
3. `*_model_destroy(handle)` exactly once

## Memory Ownership Rules

- Input tensors (`x`, `y`) are caller-owned contiguous typed arrays.
- Output tensors (`out`) are caller-owned typed arrays preallocated to required size.
- Native code does not own caller buffers and must never free them.
- Native model state is owned by Zig and released only via `*_model_destroy`.

## Tensor Layout

- `x` must be row-major contiguous `Float64Array` with shape `[n_samples, n_features]`.
- `y` is contiguous `Float64Array` (`LinearRegression`/`LogisticRegression`) or `Uint8Array` for classifier labels where required.

## Runtime Bridges

- Bun FFI bridge: `src/native/zigKernels.ts` (`bun:ffi`).
- Node-API bridge addon: `src/native/node-addon/bun_scikit_addon.cpp`.

Environment controls:

- `BUN_SCIKIT_NATIVE_BRIDGE=node-api|ffi` (default tries Node-API then FFI)
- `BUN_SCIKIT_NODE_ADDON=/absolute/path/to/bun_scikit_node_addon.node`
- `BUN_SCIKIT_ZIG_LIB=/absolute/path/to/bun_scikit_kernels.<ext>`
