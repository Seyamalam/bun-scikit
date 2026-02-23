# bun-scikit

[![CI](https://github.com/Seyamalam/bun-scikit/actions/workflows/ci.yml/badge.svg)](https://github.com/Seyamalam/bun-scikit/actions/workflows/ci.yml)
[![Benchmark Snapshot](https://github.com/Seyamalam/bun-scikit/actions/workflows/benchmark-snapshot.yml/badge.svg)](https://github.com/Seyamalam/bun-scikit/actions/workflows/benchmark-snapshot.yml)

`bun-scikit` is a scikit-learn-inspired machine learning library for Bun + TypeScript.

## Features

- `StandardScaler`
- `LinearRegression` (`normal` and `gd` solvers)
- `LogisticRegression` (binary classification, optional Zig backend)
- `KNeighborsClassifier`
- `DecisionTreeClassifier`
- `RandomForestClassifier`
- `trainTestSplit`
- Regression metrics: `meanSquaredError`, `meanAbsoluteError`, `r2Score`
- Classification metrics: `accuracyScore`, `precisionScore`, `recallScore`, `f1Score`
- Dataset-driven benchmark and CI comparison against Python `scikit-learn`

`test_data/heart.csv` is used for integration testing and benchmark comparison.

## Native Acceleration (Optional)

`LinearRegression` (`solver: "normal"`) and `LogisticRegression` support a native Zig backend.

```bash
bun run native:build
```

```ts
const linear = new LinearRegression({ solver: "normal", backend: "auto" });
const logistic = new LogisticRegression({ backend: "auto" });

linear.fit(XTrain, yTrain);
logistic.fit(XTrain, yTrain);
console.log(linear.fitBackend_, linear.fitBackendLibrary_);
console.log(logistic.fitBackend_, logistic.fitBackendLibrary_);
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
| bun-scikit | StandardScaler + LinearRegression(normal) | 0.2751 | 0.0514 | 0.117545 | 0.529539 |
| python-scikit-learn | StandardScaler + LinearRegression | 0.6437 | 0.0824 | 0.117545 | 0.529539 |

Bun fit speedup vs scikit-learn: 2.340x
Bun predict speedup vs scikit-learn: 1.603x
MSE delta (bun - sklearn): 6.362e-14
R2 delta (bun - sklearn): -2.540e-13

### Classification

| Implementation | Model | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LogisticRegression(gd,zig) | 2.0931 | 0.0544 | 0.863415 | 0.876106 |
| python-scikit-learn | StandardScaler + LogisticRegression(lbfgs) | 1.9931 | 0.1261 | 0.863415 | 0.875000 |

Bun fit speedup vs scikit-learn: 0.952x
Bun predict speedup vs scikit-learn: 2.319x
Accuracy delta (bun - sklearn): 0.000e+0
F1 delta (bun - sklearn): 1.106e-3

### Tree Classification

| Model | Implementation | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) | bun-scikit | 1.4751 | 0.0214 | 0.946341 | 0.948837 |
| DecisionTreeClassifier | python-scikit-learn | 1.8485 | 0.1343 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | bun-scikit | 43.8214 | 1.6327 | 0.990244 | 0.990566 |
| RandomForestClassifier | python-scikit-learn | 109.5692 | 6.8000 | 0.995122 | 0.995261 |

DecisionTree fit speedup vs scikit-learn: 1.253x
DecisionTree predict speedup vs scikit-learn: 6.275x
DecisionTree accuracy delta (bun - sklearn): 1.463e-2
DecisionTree f1 delta (bun - sklearn): 1.487e-2

RandomForest fit speedup vs scikit-learn: 2.500x
RandomForest predict speedup vs scikit-learn: 4.165x
RandomForest accuracy delta (bun - sklearn): -4.878e-3
RandomForest f1 delta (bun - sklearn): -4.695e-3

Snapshot generated at: 2026-02-23T09:54:58.330Z
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
