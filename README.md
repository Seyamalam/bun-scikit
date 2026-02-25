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
- Baselines: `DummyClassifier`, `DummyRegressor`.
- Preprocessing: `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `MaxAbsScaler`, `Normalizer`, `Binarizer`, `LabelEncoder`, `PolynomialFeatures`, `SimpleImputer`, `OneHotEncoder`.
- Composition: `Pipeline`, `ColumnTransformer`, `FeatureUnion`.
- Feature selection: `VarianceThreshold`.
- Model selection: `trainTestSplit`, `KFold`, stratified/repeated splitters, `crossValScore`, `GridSearchCV`, `RandomizedSearchCV`.
- Metrics: regression and classification metrics, including `logLoss`, `rocAucScore`, `confusionMatrix`, `classificationReport`, `balancedAccuracyScore`, `matthewsCorrcoef`, `brierScoreLoss`, `meanAbsolutePercentageError`, and `explainedVarianceScore`.

## Scikit Parity Matrix

| Area | Status |
| --- | --- |
| Linear models | `LinearRegression`, `LogisticRegression`, `SGDClassifier`, `SGDRegressor`, `LinearSVC` |
| Tree/ensemble | `DecisionTreeClassifier`, `DecisionTreeRegressor`, `RandomForestClassifier`, `RandomForestRegressor` |
| Neighbors / Bayes | `KNeighborsClassifier`, `GaussianNB` |
| Baselines | `DummyClassifier`, `DummyRegressor` |
| Preprocessing | `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `MaxAbsScaler`, `Normalizer`, `Binarizer`, `LabelEncoder`, `PolynomialFeatures`, `SimpleImputer`, `OneHotEncoder` |
| Feature selection | `VarianceThreshold` |
| Model selection | `trainTestSplit`, `KFold`, `StratifiedKFold`, `StratifiedShuffleSplit`, `RepeatedKFold`, `RepeatedStratifiedKFold`, `crossValScore`, `GridSearchCV`, `RandomizedSearchCV` |
| Metrics (regression) | `meanSquaredError`, `meanAbsoluteError`, `r2Score`, `meanAbsolutePercentageError`, `explainedVarianceScore` |
| Metrics (classification) | `accuracyScore`, `precisionScore`, `recallScore`, `f1Score`, `balancedAccuracyScore`, `matthewsCorrcoef`, `logLoss`, `brierScoreLoss`, `rocAucScore`, `confusionMatrix`, `classificationReport` |

Near-term parity gaps vs scikit-learn include clustering, decomposition, calibration, advanced feature selection, and probability calibration/meta-estimators.

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
- `BUN_SCIKIT_TREE_BACKEND=zig|js` (default is `zig`; set `js` to force JS tree/forest fallback)

## Performance Snapshot

Latest CI snapshot on `test_data/heart.csv` vs Python scikit-learn:

- Regression: fit `1.904x`, predict `2.187x` (MSE delta `6.362e-14`, R2 delta `-2.539e-13`)
- Classification: fit `2.033x`, predict `2.548x` (accuracy delta `0.000e+0`, F1 delta `1.106e-3`)
- DecisionTree (`js-fast`): fit `1.512x`, predict `4.419x`
- RandomForest (`js-fast`): fit `2.082x`, predict `1.065x`
- Tree backend matrix: DecisionTree `zig/js` fit `1.958x`, predict `0.550x`; RandomForest `zig/js` fit `2.829x`, predict `2.392x`
- Snapshot generated at `2026-02-25T19:23:37.811Z`
- Tree backend matrix (`js-fast` vs `zig-tree` vs `sklearn`) is included in `bench/results/heart-ci-latest.md`
- Synthetic tree/forest hot-path benchmark command: `bun run bench:hotpaths`

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
