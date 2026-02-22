# bun-scikit

[![CI](https://github.com/Seyamalam/bun-scikit/actions/workflows/ci.yml/badge.svg)](https://github.com/Seyamalam/bun-scikit/actions/workflows/ci.yml)
[![Benchmark Snapshot](https://github.com/Seyamalam/bun-scikit/actions/workflows/benchmark-snapshot.yml/badge.svg)](https://github.com/Seyamalam/bun-scikit/actions/workflows/benchmark-snapshot.yml)

`bun-scikit` is a scikit-learn-inspired machine learning library for Bun + TypeScript.

## Features

- `StandardScaler`
- `LinearRegression` (`normal` and `gd` solvers)
- `LogisticRegression` (binary classification, optional Zig fit backend)
- `KNeighborsClassifier`
- `DecisionTreeClassifier`
- `RandomForestClassifier`
- `trainTestSplit`
- Regression metrics: `meanSquaredError`, `meanAbsoluteError`, `r2Score`
- Classification metrics: `accuracyScore`, `precisionScore`, `recallScore`, `f1Score`
- Dataset-driven benchmark and CI comparison against Python `scikit-learn`

`test_data/heart.csv` is used for integration testing and benchmark comparison.

## Native Acceleration (Optional)

`LogisticRegression` supports a native Zig backend for faster fit epochs.

```bash
bun run native:build
```

```ts
const model = new LogisticRegression({ backend: "auto" });
model.fit(XTrain, yTrain);
console.log(model.fitBackend_, model.fitBackendLibrary_);
```

Backends:

- `auto` (default): use Zig if found, otherwise JS fallback
- `js`: force JavaScript/TypeScript path
- `zig`: require native kernel (throws if missing)

## Install

```bash
bun install bun-scikit
```

## Usage

```ts
import {
  LinearRegression,
  StandardScaler,
  meanSquaredError,
  trainTestSplit,
} from "bun-scikit";

const X = [
  [1, 2],
  [2, 3],
  [3, 4],
  [4, 5],
];
const y = [5, 7, 9, 11];

const scaler = new StandardScaler();
const XScaled = scaler.fitTransform(X);
const { XTrain, XTest, yTrain, yTest } = trainTestSplit(XScaled, y, {
  testSize: 0.25,
  randomState: 42,
});

const model = new LinearRegression({ solver: "normal" });
model.fit(XTrain, yTrain);
const predictions = model.predict(XTest);

console.log("MSE:", meanSquaredError(yTest, predictions));
```

## Benchmarks

The table below is generated from `bench/results/heart-ci-latest.json`.
That snapshot is produced by CI in `.github/workflows/benchmark-snapshot.yml`.

<!-- BENCHMARK_TABLE_START -->
Benchmark snapshot source: `bench/results/heart-ci-latest.json` (generated in CI workflow `Benchmark Snapshot`).
Dataset: `test_data/heart.csv` (1025 samples, 13 features, test fraction 0.2).

### Regression

| Implementation | Model | Fit median (ms) | Predict median (ms) | MSE | R2 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LinearRegression(normal) | 0.2690 | 0.0147 | 0.117545 | 0.529539 |
| python-scikit-learn | StandardScaler + LinearRegression | 0.3004 | 0.0357 | 0.117545 | 0.529539 |

Bun fit speedup vs scikit-learn: 1.117x
Bun predict speedup vs scikit-learn: 2.425x
MSE delta (bun - sklearn): 6.362e-14
R2 delta (bun - sklearn): -2.539e-13

### Classification

| Implementation | Model | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LogisticRegression(gd,zig) | 1.0356 | 0.0195 | 0.863415 | 0.876106 |
| python-scikit-learn | StandardScaler + LogisticRegression(lbfgs) | 1.0588 | 0.0702 | 0.863415 | 0.875000 |

Bun fit speedup vs scikit-learn: 1.023x
Bun predict speedup vs scikit-learn: 3.588x
Accuracy delta (bun - sklearn): 0.000e+0
F1 delta (bun - sklearn): 1.106e-3

### Tree Classification

| Model | Implementation | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) | bun-scikit | 1.0874 | 0.0176 | 0.946341 | 0.948837 |
| DecisionTreeClassifier | python-scikit-learn | 1.2841 | 0.0850 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | bun-scikit | 32.1913 | 1.4896 | 0.990244 | 0.990566 |
| RandomForestClassifier | python-scikit-learn | 65.2060 | 2.0660 | 0.995122 | 0.995261 |

DecisionTree fit speedup vs scikit-learn: 1.181x
DecisionTree predict speedup vs scikit-learn: 4.832x
DecisionTree accuracy delta (bun - sklearn): 1.463e-2
DecisionTree f1 delta (bun - sklearn): 1.487e-2

RandomForest fit speedup vs scikit-learn: 2.026x
RandomForest predict speedup vs scikit-learn: 1.387x
RandomForest accuracy delta (bun - sklearn): -4.878e-3
RandomForest f1 delta (bun - sklearn): -4.695e-3

Snapshot generated at: 2026-02-22T17:19:58.072Z
<!-- BENCHMARK_TABLE_END -->

## Documentation

- Docs index: `docs/README.md`
- Getting started: `docs/getting-started.md`
- API reference: `docs/api.md`
- Benchmarking flow: `docs/benchmarking.md`
- Zig acceleration: `docs/zig-acceleration.md`

## Maintainer Files

- Changelog: `CHANGELOG.md`
- Contributing guide: `CONTRIBUTING.md`
- Code of Conduct: `CODE_OF_CONDUCT.md`
- Security policy: `SECURITY.md`
- Support policy: `SUPPORT.md`
- License: `LICENSE`

## Local Commands

```bash
bun run test
bun run typecheck
bun run docs:api:generate
bun run docs:coverage:check
bun run bench
bun run bench:heart:classification
bun run bench:heart:tree
bun run bench:ci
bun run bench:ci:native
bun run bench:snapshot
bun run native:build
```
