# Zig Acceleration

`bun-scikit` uses a Zig backend for compute-heavy training loops.
Current coverage:

- `LinearRegression` (`solver: "normal"`) native model handle (`fit` + `predict`)
- `LogisticRegression` native model handle (`fit` + `predict` / `predictProba`)

## Build Native Kernels

```bash
bun run native:build
```

This generates:

- `dist/native/bun_scikit_kernels.so` (Linux)
- `dist/native/bun_scikit_kernels.dylib` (macOS)
- `dist/native/bun_scikit_kernels.dll` (Windows)

## Use In Code

```ts
import { LinearRegression, LogisticRegression } from "bun-scikit";

const linear = new LinearRegression({
  solver: "normal",
});

const logistic = new LogisticRegression();
```

Behavior:

- Native Zig kernels are required for `LinearRegression.fit()` and `LogisticRegression.fit()`.
- If kernels are unavailable, `fit()` throws and asks you to run `bun run native:build`.

After `fit`, inspect:

- `fitBackend_` (`"zig"`)
- `fitBackendLibrary_` (native library path when Zig is used)

## Environment Variables

- `BUN_SCIKIT_ENABLE_ZIG=0` disables Zig backend discovery globally.
- `BUN_SCIKIT_ZIG_LIB=/absolute/path/to/bun_scikit_kernels.<ext>` forces a specific native library path.
- `BUN_SCIKIT_NATIVE_BRIDGE=node-api|ffi` controls bridge choice (`node-api` preferred when present).
- `BUN_SCIKIT_NODE_ADDON=/absolute/path/to/bun_scikit_node_addon.node` forces a specific Node-API addon path.

## Build Node-API Bridge (Experimental)

```bash
bun run native:build:node-addon
```

Or build both:

```bash
bun run native:build:all
```

## Benchmarks

CI benchmark workflows compile native Zig kernels before running the Bun vs
scikit-learn snapshots, so published benchmark tables include the accelerated path.
