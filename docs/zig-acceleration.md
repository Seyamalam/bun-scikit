# Zig Acceleration

`bun-scikit` supports an optional Zig backend for compute-heavy training loops.
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
  backend: "auto", // "auto" | "js" | "zig"
});

const logistic = new LogisticRegression({
  backend: "auto", // "auto" | "js" | "zig"
});
```

Behavior:

- `backend: "auto"`: use Zig if a native kernel is found, else JS fallback.
- `backend: "js"`: always use pure TypeScript/JavaScript.
- `backend: "zig"`: require native kernels; throws if unavailable.

After `fit`, inspect:

- `fitBackend_` (`"js"` or `"zig"`)
- `fitBackendLibrary_` (native library path when Zig is used)

## Environment Variables

- `BUN_SCIKIT_ENABLE_ZIG=0` disables Zig backend discovery globally.
- `BUN_SCIKIT_ZIG_LIB=/absolute/path/to/bun_scikit_kernels.<ext>` forces a specific native library path.

## Benchmarks

CI benchmark workflows compile native Zig kernels before running the Bun vs
scikit-learn snapshots, so published benchmark tables include the accelerated path.
