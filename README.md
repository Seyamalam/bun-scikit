# bun-scikit

[![CI](https://github.com/Seyamalam/bun-scikit/actions/workflows/ci.yml/badge.svg)](https://github.com/Seyamalam/bun-scikit/actions/workflows/ci.yml)
[![Benchmark Snapshot](https://github.com/Seyamalam/bun-scikit/actions/workflows/benchmark-snapshot.yml/badge.svg)](https://github.com/Seyamalam/bun-scikit/actions/workflows/benchmark-snapshot.yml)

Scikit-learn-inspired machine learning for Bun + TypeScript, with native Zig acceleration for core training paths.

## Install

```bash
bun add bun-scikit
```

## Verify Zig Backend (Post-Install Smoke Test)

Create `index.ts`:

```ts
import { DecisionTreeClassifier, RandomForestClassifier } from "bun-scikit";

const X = [
  [0, 0],
  [0, 1],
  [1, 0],
  [2, 2],
  [2, 3],
  [3, 2],
];
const y = [0, 0, 0, 1, 1, 1];

const tree = new DecisionTreeClassifier({ maxDepth: 3, randomState: 42 });
tree.fit(X, y);
console.log("DecisionTree fit backend:", tree.fitBackend_, tree.fitBackendLibrary_);

const forest = new RandomForestClassifier({ nEstimators: 25, maxDepth: 4, randomState: 42 });
forest.fit(X, y);
console.log("RandomForest fit backend:", forest.fitBackend_, forest.fitBackendLibrary_);
```

Run:

```bash
bun run index.ts
```

Expected output includes `fit backend: zig` for both models.

Repo example: `examples/zig-backend-smoke.ts`

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

- Models: `LinearRegression`, `LogisticRegression`, `LogisticRegressionCV`, `Ridge`, `Lasso`, `ElasticNet`, `RidgeCV`, `LassoCV`, `ElasticNetCV`, `BayesianRidge`, `ARDRegression`, `Perceptron`, `PassiveAggressiveClassifier`, `PassiveAggressiveRegressor`, `HuberRegressor`, `PoissonRegressor`, `GammaRegressor`, `QuantileRegressor`, `MultiTaskLasso`, `MultiTaskElasticNet`, `MultiTaskLassoCV`, `MultiTaskElasticNetCV`, `KNeighborsClassifier`, `DecisionTreeClassifier`, `RandomForestClassifier`, plus additional parity models (`LinearSVC`, `GaussianNB`, `SGDClassifier`, `SGDRegressor`, regressors for tree/forest, `OneClassSVM`).
- Clustering / decomposition / manifold: `KMeans`, `MiniBatchKMeans`, `DBSCAN`, `AgglomerativeClustering`, `SpectralClustering`, `Birch`, `OPTICS`, `MeanShift`, `AffinityPropagation`, `PCA`, `IncrementalPCA`, `TruncatedSVD`, `FastICA`, `NMF`, `MiniBatchNMF`, `FactorAnalysis`, `KernelPCA`, `PLSSVD`, `PLSRegression`, `PLSCanonical`, `CCA`, `TSNE`, `Isomap`, `LocallyLinearEmbedding`, `MDS`.
- Anomaly detection: `IsolationForest`, `LocalOutlierFactor`, `OneClassSVM`.
- Calibration / meta-estimators: `CalibratedClassifierCV`, `VotingClassifier`, `VotingRegressor`, `StackingClassifier`, `StackingRegressor`, `BaggingClassifier`.
- Gaussian process / isotonic: `GaussianProcessRegressor`, `GaussianProcessClassifier`, `IsotonicRegression`.
- Multioutput: `MultiOutputClassifier`, `MultiOutputRegressor`, `ClassifierChain`, `RegressorChain`.
- Boosting: `AdaBoostClassifier`, `GradientBoostingClassifier`, `GradientBoostingRegressor`, `HistGradientBoostingClassifier`, `HistGradientBoostingRegressor`.
- Baselines: `DummyClassifier`, `DummyRegressor`.
- Neighbors / Bayes additions: `BallTree`, `KDTree`, `KNeighborsTransformer`, `NearestCentroid`, `NeighborhoodComponentsAnalysis`, `BernoulliNB`, `MultinomialNB`, `ComplementNB`, `CategoricalNB`.
- Covariance additions: `EllipticEnvelope`, `GraphicalLasso`, `GraphicalLassoCV`.
- Preprocessing: `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `MaxAbsScaler`, `Normalizer`, `Binarizer`, `LabelEncoder`, `LabelBinarizer`, `MultiLabelBinarizer`, `PolynomialFeatures`, `SimpleImputer`, `OneHotEncoder`, `FunctionTransformer`, `KernelCenterer`.
- Feature extraction: `DictVectorizer`, `FeatureHasher`.
- Composition: `Pipeline`, `ColumnTransformer`, `FeatureUnion`.
- Feature selection: `VarianceThreshold`, `SelectKBest`, `SelectPercentile`, `SelectFromModel`, `RFE`, `RFECV`, `chi2`, `f_classif`, `f_regression`, `mutualInfoClassif`, `mutualInfoRegression`.
- Model selection: `trainTestSplit`, `KFold`, `GroupKFold`, `GroupShuffleSplit`, `StratifiedKFold`, `StratifiedGroupKFold`, stratified/repeated splitters, `crossValScore`, `crossValidate`, `crossValPredict`, `learningCurve`, `validationCurve`, `GridSearchCV`, `RandomizedSearchCV`, `ParameterGrid`, `ParameterSampler`.
- Metrics: regression/classification metrics plus clustering metrics (`silhouetteScore`, `calinskiHarabaszScore`, `daviesBouldinScore`, `adjustedRandScore`).
- Inspection: `permutationImportance`, `partialDependence`, `permutationTestScore`.

## Scikit Parity Matrix

| Area | Status |
| --- | --- |
| Linear models | `LinearRegression`, `LogisticRegression`, `SGDClassifier`, `SGDRegressor`, `LinearSVC` |
| Tree/ensemble | `DecisionTreeClassifier`, `DecisionTreeRegressor`, `RandomForestClassifier`, `RandomForestRegressor`, `AdaBoostClassifier`, `GradientBoostingClassifier`, `GradientBoostingRegressor`, `HistGradientBoostingClassifier`, `HistGradientBoostingRegressor` |
| Neighbors / Bayes | `KNeighborsClassifier`, `KNeighborsRegressor`, `GaussianNB` |
| Clustering | `KMeans`, `MiniBatchKMeans`, `DBSCAN`, `AgglomerativeClustering`, `SpectralClustering`, `Birch`, `OPTICS`, `MeanShift`, `AffinityPropagation` |
| Decomposition / Manifold | `PCA`, `IncrementalPCA`, `TruncatedSVD`, `FastICA`, `NMF`, `MiniBatchNMF`, `FactorAnalysis`, `KernelPCA`, `PLSSVD`, `PLSRegression`, `PLSCanonical`, `CCA`, `TSNE`, `Isomap`, `LocallyLinearEmbedding`, `MDS` |
| Anomaly detection | `IsolationForest`, `LocalOutlierFactor`, `OneClassSVM` |
| Calibration / Meta | `CalibratedClassifierCV`, `VotingClassifier`, `VotingRegressor`, `StackingClassifier`, `StackingRegressor`, `BaggingClassifier`, `BaggingRegressor`, `OneVsRestClassifier`, `OneVsOneClassifier` |
| Multioutput | `MultiOutputClassifier`, `MultiOutputRegressor`, `ClassifierChain`, `RegressorChain` |
| Baselines | `DummyClassifier`, `DummyRegressor` |
| Preprocessing | `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `MaxAbsScaler`, `Normalizer`, `Binarizer`, `LabelEncoder`, `PolynomialFeatures`, `SimpleImputer`, `OneHotEncoder` |
| Feature selection | `VarianceThreshold`, `SelectKBest`, `SelectPercentile`, `SelectFromModel`, `RFE`, `RFECV`, `chi2`, `f_classif`, `f_regression`, `mutualInfoClassif`, `mutualInfoRegression` |
| Model selection | `trainTestSplit`, `KFold`, `StratifiedKFold`, `StratifiedShuffleSplit`, `RepeatedKFold`, `RepeatedStratifiedKFold`, `crossValScore`, `crossValidate`, `crossValPredict`, `learningCurve`, `validationCurve`, `GridSearchCV`, `RandomizedSearchCV`, `ParameterGrid`, `ParameterSampler` |
| Metrics (regression) | `meanSquaredError`, `meanAbsoluteError`, `r2Score`, `meanAbsolutePercentageError`, `explainedVarianceScore` |
| Metrics (classification) | `accuracyScore`, `precisionScore`, `recallScore`, `f1Score`, `balancedAccuracyScore`, `matthewsCorrcoef`, `logLoss`, `brierScoreLoss`, `rocAucScore`, `confusionMatrix`, `classificationReport` |
| Metrics (clustering) | `silhouetteScore`, `calinskiHarabaszScore`, `daviesBouldinScore`, `adjustedRandScore` |
| Inspection | `permutationImportance`, `partialDependence`, `permutationTestScore` |

### Parity Coverage vs README

Parity status is aligned across runtime exports, matrix contracts, and docs coverage.

Source of required runtime surface: `docs/parity-matrix.json` (`209` total runtime exports).

Tracked parity status (latest check):
- API surface parity: `209 / 209` required exports (`100%`).
- API/class/interface contract parity: `0` failures (`100%` pass).
- API docs coverage: `470 / 470` exported symbols referenced in `docs/api.md` (`100%`).
- sklearn snapshot parity gate metrics: `34 / 34` pass (`100%`).
- Full sklearn public-symbol coverage (non-strict inventory gate): `203 / 454` (`44.71%`).

Artifacts:
- `bench/results/parity/parity-report-latest.md`
- `bench/results/parity/parity-matrix-report.json`
- `bench/results/parity/parity-sklearn-report.json`
- `bench/results/parity/parity-full-report.json`
- `docs/sklearn-public-api.json`

Commands:
- Regenerate sklearn inventory: `bun run parity:inventory:generate`
- Check runtime export + contract matrix parity: `bun run parity:matrix:check`
- Check docs symbol coverage: `bun run docs:coverage:check`
- Check full symbol coverage (report only): `bun run parity:full:check`
- Enforce strict full-symbol gate: `PARITY_FULL_STRICT=1 bun run parity:full:check`

Beyond the tracked matrix, remaining gaps to full scikit-learn-wide one-to-one behavior are mainly untracked modules and APIs, including:
- Feature extraction families (for example text/image vectorizers and hashing/vectorization utilities).
- Additional decomposition/manifold variants and solvers not currently exposed.
- Additional covariance and gaussian-process families.
- Additional inspection/display/reporting utilities and plotting-oriented helpers.
- Dataset utilities and other sklearn ecosystem helpers outside this runtime-focused library surface.

Multiclass support is available for `GaussianNB`, `KNeighborsClassifier`, `LogisticRegression`, `SGDClassifier`, `LinearSVC`, `DecisionTreeClassifier`, `RandomForestClassifier`, `VotingClassifier`, `StackingClassifier`, `BaggingClassifier`, and `CalibratedClassifierCV`.

`DecisionTreeClassifier` and `RandomForestClassifier` now support multiclass native Zig fit/predict paths (up to 256 encoded classes) when `BUN_SCIKIT_TREE_BACKEND=zig`.

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

Parity checks are enforced in CI using:
- API parity matrix coverage (`bun run parity:matrix:check`)
- sklearn snapshot fixtures with multi-seed drift checks (`bun run parity:check`)
- versioned parity report artifacts (`bun run parity:report`)

## Performance Snapshot

<!-- BENCHMARK_TABLE_START -->
Benchmark snapshot source: `bench/results/heart-ci-latest.json` (generated in CI workflow `Benchmark Snapshot`).
Dataset: `test_data/heart.csv` (1025 samples, 13 features, test fraction 0.2).

### Summary

- Regression: fit `2.204x`, predict `2.430x` (MSE delta `6.362e-14`, R2 delta `-2.539e-13`)
- Classification: fit `2.452x`, predict `2.601x` (accuracy delta `0.000e+0`, F1 delta `1.106e-3`)
- DecisionTree (`js-fast`): fit `1.645x`, predict `4.438x`
- RandomForest (`js-fast`): fit `6.395x`, predict `3.924x`
- Tree backend matrix: DecisionTree `zig/js` fit `1.819x`, predict `0.617x`; RandomForest `zig/js` fit `2.650x`, predict `2.256x`
- Snapshot generated at `2026-02-25T19:47:51.136Z`

### Regression

| Implementation | Model | Fit median (ms) | Predict median (ms) | MSE | R2 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LinearRegression(normal) | 0.1763 | 0.0186 | 0.117545 | 0.529539 |
| python-scikit-learn | StandardScaler + LinearRegression | 0.3886 | 0.0452 | 0.117545 | 0.529539 |

Bun fit speedup vs scikit-learn: 2.204x
Bun predict speedup vs scikit-learn: 2.430x
MSE delta (bun - sklearn): 6.362e-14
R2 delta (bun - sklearn): -2.539e-13

### Classification

| Implementation | Model | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LogisticRegression(gd,zig) | 0.5275 | 0.0320 | 0.863415 | 0.876106 |
| python-scikit-learn | StandardScaler + LogisticRegression(lbfgs) | 1.2934 | 0.0833 | 0.863415 | 0.875000 |

Bun fit speedup vs scikit-learn: 2.452x
Bun predict speedup vs scikit-learn: 2.601x
Accuracy delta (bun - sklearn): 0.000e+0
F1 delta (bun - sklearn): 1.106e-3

### Tree Classification

| Model | Implementation | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) [js-fast] | bun-scikit | 0.8338 | 0.0209 | 0.946341 | 0.948837 |
| DecisionTreeClassifier | python-scikit-learn | 1.3712 | 0.0928 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) [js-fast] | bun-scikit | 31.2166 | 1.7649 | 0.990244 | 0.990566 |
| RandomForestClassifier | python-scikit-learn | 199.6324 | 6.9251 | 0.995122 | 0.995261 |

DecisionTree fit speedup vs scikit-learn: 1.645x
DecisionTree predict speedup vs scikit-learn: 4.438x
DecisionTree accuracy delta (bun - sklearn): 1.463e-2
DecisionTree f1 delta (bun - sklearn): 1.487e-2

RandomForest fit speedup vs scikit-learn: 6.395x
RandomForest predict speedup vs scikit-learn: 3.924x
RandomForest accuracy delta (bun - sklearn): -4.878e-3
RandomForest f1 delta (bun - sklearn): -4.695e-3

### Tree Backend Modes (Bun vs Bun vs sklearn)

| Model | Backend | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) | js-fast | 0.8338 | 0.0209 | 0.946341 | 0.948837 |
| DecisionTreeClassifier(maxDepth=8) | zig-tree | 0.4583 | 0.0339 | 0.892683 | 0.899083 |
| DecisionTreeClassifier | python-scikit-learn | 1.3712 | 0.0928 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | js-fast | 31.2166 | 1.7649 | 0.990244 | 0.990566 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | zig-tree | 11.7783 | 0.7824 | 0.995122 | 0.995261 |
| RandomForestClassifier | python-scikit-learn | 199.6324 | 6.9251 | 0.995122 | 0.995261 |

DecisionTree zig/js fit speedup: 1.819x
DecisionTree zig/js predict speedup: 0.617x
RandomForest zig/js fit speedup: 2.650x
RandomForest zig/js predict speedup: 2.256x

Snapshot generated at: 2026-02-25T19:47:51.136Z
<!-- BENCHMARK_TABLE_END -->

## Documentation

- Getting started: `docs/getting-started.md`
- API reference: `docs/api.md`
- Benchmarking: `docs/benchmarking.md`
- Zig acceleration: `docs/zig-acceleration.md`
- Native ABI: `docs/native-abi.md`
- Release checklist: `docs/release-checklist.md`
- Release notes draft automation: `bun run release:notes` (updates `docs/release-notes/v*.md` + parity block in `CHANGELOG.md`)

## Contributing / Project Files

- Changelog: `CHANGELOG.md`
- Contributing: `CONTRIBUTING.md`
- Security: `SECURITY.md`
- Code of Conduct: `CODE_OF_CONDUCT.md`
- Support: `SUPPORT.md`
