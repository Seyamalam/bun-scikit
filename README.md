# bun-scikit

[![CI](https://github.com/Seyamalam/bun-scikit/actions/workflows/ci.yml/badge.svg)](https://github.com/Seyamalam/bun-scikit/actions/workflows/ci.yml)
[![Benchmark Snapshot](https://github.com/Seyamalam/bun-scikit/actions/workflows/benchmark-snapshot.yml/badge.svg)](https://github.com/Seyamalam/bun-scikit/actions/workflows/benchmark-snapshot.yml)

`bun-scikit` is a scikit-learn-inspired machine learning library for Bun + TypeScript.

## Features

- `StandardScaler`
- `LinearRegression` (`normal` and `gd` solvers)
- `LogisticRegression` (binary classification)
- `KNeighborsClassifier`
- `DecisionTreeClassifier`
- `RandomForestClassifier`
- `trainTestSplit`
- Regression metrics: `meanSquaredError`, `meanAbsoluteError`, `r2Score`
- Classification metrics: `accuracyScore`, `precisionScore`, `recallScore`, `f1Score`
- Dataset-driven benchmark and CI comparison against Python `scikit-learn`

`test_data/heart.csv` is used for integration testing and benchmark comparison.

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
| bun-scikit | StandardScaler + LinearRegression(normal) | 0.7298 | 0.0243 | 0.117545 | 0.529539 |
| python-scikit-learn | StandardScaler + LinearRegression | 1.1323 | 0.1461 | 0.117545 | 0.529539 |

Bun fit speedup vs scikit-learn: 1.552x
Bun predict speedup vs scikit-learn: 6.005x
MSE delta (bun - sklearn): 6.363e-14
R2 delta (bun - sklearn): -2.540e-13

### Classification

| Implementation | Model | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LogisticRegression(gd) | 16.9612 | 0.0394 | 0.863415 | 0.875000 |
| python-scikit-learn | StandardScaler + LogisticRegression(lbfgs) | 3.5231 | 0.1938 | 0.863415 | 0.875000 |

Bun fit speedup vs scikit-learn: 0.208x
Bun predict speedup vs scikit-learn: 4.915x
Accuracy delta (bun - sklearn): 0.000e+0
F1 delta (bun - sklearn): -1.110e-16

### Tree Classification

| Model | Implementation | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) | bun-scikit | 13.8622 | 0.0300 | 0.931707 | 0.935185 |
| DecisionTreeClassifier | python-scikit-learn | 2.4583 | 0.1800 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | bun-scikit | 225.1871 | 1.7256 | 0.995122 | 0.995261 |
| RandomForestClassifier | python-scikit-learn | 110.1258 | 6.8506 | 0.995122 | 0.995261 |

DecisionTree fit speedup vs scikit-learn: 0.177x
DecisionTree predict speedup vs scikit-learn: 6.005x
DecisionTree accuracy delta (bun - sklearn): 0.000e+0
DecisionTree f1 delta (bun - sklearn): 1.223e-3

RandomForest fit speedup vs scikit-learn: 0.489x
RandomForest predict speedup vs scikit-learn: 3.970x
RandomForest accuracy delta (bun - sklearn): 0.000e+0
RandomForest f1 delta (bun - sklearn): 1.110e-16

Snapshot generated at: 2026-02-22T12:17:31.022Z
<!-- BENCHMARK_TABLE_END -->

## Documentation

- Docs index: `docs/README.md`
- Getting started: `docs/getting-started.md`
- API reference: `docs/api.md`
- Benchmarking flow: `docs/benchmarking.md`

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
bun run bench:snapshot
```
