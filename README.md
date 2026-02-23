# bun-scikit

[![CI](https://github.com/Seyamalam/bun-scikit/actions/workflows/ci.yml/badge.svg)](https://github.com/Seyamalam/bun-scikit/actions/workflows/ci.yml)
[![Benchmark Snapshot](https://github.com/Seyamalam/bun-scikit/actions/workflows/benchmark-snapshot.yml/badge.svg)](https://github.com/Seyamalam/bun-scikit/actions/workflows/benchmark-snapshot.yml)

Scikit-learn-inspired machine learning for Bun + TypeScript, with native Zig acceleration for core training paths.

## Install

```bash
bun add bun-scikit
```

## Quick Start

```ts
import {
  LinearRegression,
  LogisticRegression,
  StandardScaler,
  trainTestSplit,
  meanSquaredError,
  accuracyScore,
} from "bun-scikit";

const X = [[1], [2], [3], [4], [5], [6]];
const yReg = [3, 5, 7, 9, 11, 13];
const yCls = [0, 0, 0, 1, 1, 1];

const scaler = new StandardScaler();
const Xs = scaler.fitTransform(X);

const { XTrain, XTest, yTrain, yTest } = trainTestSplit(Xs, yReg, {
  testSize: 0.33,
  randomState: 42,
});

const reg = new LinearRegression({ solver: "normal" });
reg.fit(XTrain, yTrain);
console.log("MSE:", meanSquaredError(yTest, reg.predict(XTest)));

const clf = new LogisticRegression({
  solver: "gd",
  learningRate: 0.8,
  maxIter: 100,
  tolerance: 1e-5,
});
clf.fit(Xs, yCls);
console.log("Accuracy:", accuracyScore(yCls, clf.predict(Xs)));
```

## Included APIs

- Models: `LinearRegression`, `LogisticRegression`, `KNeighborsClassifier`, `DecisionTreeClassifier`, `RandomForestClassifier`, plus additional parity models (`LinearSVC`, `GaussianNB`, `SGDClassifier`, `SGDRegressor`, regressors for tree/forest).
- Preprocessing: `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `PolynomialFeatures`, `SimpleImputer`, `OneHotEncoder`.
- Composition: `Pipeline`, `ColumnTransformer`, `FeatureUnion`.
- Model selection: `trainTestSplit`, `KFold`, stratified/repeated splitters, `crossValScore`, `GridSearchCV`.
- Metrics: regression and classification metrics, including `logLoss`, `rocAucScore`, `confusionMatrix`, `classificationReport`.

## Native Runtime

- Prebuilt binaries are bundled in the npm package for:
  - `linux-x64`
  - `windows-x64`
- No `bun pm trust` step is required for standard install/use.
- macOS prebuilt binaries are not published yet.

Optional env vars:

- `BUN_SCIKIT_NATIVE_BRIDGE=node-api|ffi`
- `BUN_SCIKIT_NODE_ADDON=/absolute/path/to/bun_scikit_node_addon.node`
- `BUN_SCIKIT_ZIG_LIB=/absolute/path/to/bun_scikit_kernels.<ext>`

## Performance Snapshot

Latest CI snapshot on `test_data/heart.csv` vs Python scikit-learn:

- Regression: fit `1.52x`, predict `1.68x`
- Classification: fit `2.31x`, predict `2.57x`
- DecisionTree: fit `1.83x`, predict `5.24x`
- RandomForest: fit `6.26x`, predict `3.50x`

Raw benchmark artifacts:

- `bench/results/heart-ci-latest.json`
- `bench/results/heart-ci-latest.md`

## Documentation

- Getting started: `docs/getting-started.md`
- API reference: `docs/api.md`
- Benchmarking: `docs/benchmarking.md`
- Zig acceleration: `docs/zig-acceleration.md`
- Native ABI: `docs/native-abi.md`
- Release checklist: `docs/release-checklist.md`

## Contributing / Project Files

- Changelog: `CHANGELOG.md`
- Contributing: `CONTRIBUTING.md`
- Security: `SECURITY.md`
- Code of Conduct: `CODE_OF_CONDUCT.md`
- Support: `SUPPORT.md`
