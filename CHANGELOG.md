# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
<!-- PARITY_AUTO_START -->
- feat(parity): implement next sklearn parity batch (1-5)
<!-- PARITY_AUTO_END -->
- Reusable `Release Prep` workflow (`.github/workflows/release-prep.yml`) that gates release pipelines with tests, typecheck, Zig guard checks, benchmark checks, README benchmark sync checks, and npm-pack smoke validation.
- Zig backend smoke example for users: `examples/zig-backend-smoke.ts`.
- Per-kernel tree hot-path regression guard (`bench:hotpaths:check`) using `bench/results/tree-hotpaths-baseline.json`.
- `KMeans` clustering estimator with deterministic `randomState`, `fitPredict`, `transform`, and `score`.
- `PCA` decomposition estimator with explained-variance attributes, `fitTransform`, `transform`, and `inverseTransform`.
- `TruncatedSVD` decomposition estimator with deterministic power-iteration components, variance attribution, `fitTransform`, and `inverseTransform`.
- `FastICA` decomposition estimator with whitening, independent component extraction, and inverse projection support.
- `NMF` decomposition estimator with multiplicative-update factorization and reconstruction APIs.
- `KernelPCA` decomposition estimator (`linear`, `rbf`, `poly` kernels).
- `DBSCAN` and `AgglomerativeClustering` for density-based and hierarchical clustering parity.
- `CalibratedClassifierCV` with `sigmoid` and `isotonic` calibration methods.
- `VotingClassifier` (hard/soft) and `StackingClassifier` baseline meta-estimators.
- `VotingRegressor`, `StackingRegressor`, and `BaggingClassifier` meta-estimator parity additions.
- Boosting estimators: `AdaBoostClassifier`, `GradientBoostingClassifier`, and `GradientBoostingRegressor`.
- Histogram boosting estimators: `HistGradientBoostingClassifier` and `HistGradientBoostingRegressor`.
- Multiclass support across core classifiers/ensembles (`GaussianNB`, `KNeighborsClassifier`, `LogisticRegression`, `SGDClassifier`, `LinearSVC`, `DecisionTreeClassifier`, `RandomForestClassifier`, `VotingClassifier`, `StackingClassifier`, `BaggingClassifier`, `CalibratedClassifierCV`).
- sklearn snapshot fixtures at `test/fixtures/sklearn-snapshots.json` and fixture-based parity tests for calibration/ensemble/decomposition outputs.
- Reproducible fixture generation script: `scripts/generate-sklearn-fixtures.py`.
- Feature-importance API (`featureImportances_`) for tree/forest/boosting estimators, including histogram boosting.
- `Release Guard` workflow (`.github/workflows/release-guard.yml`) with tag/version/npm preflight checks and duplicate publish/native-run detection.
- Concurrency dedupe guards on publish workflows to avoid duplicate tag releases.

### Changed
- README install docs now include a post-install Zig backend smoke check for `DecisionTreeClassifier` and `RandomForestClassifier`.
- README benchmark section is now marker-driven and auto-generated from `bench/results/heart-ci-latest.json` via `scripts/sync-benchmark-readme.ts`.
- `bench:snapshot` now also runs `bench:sync-readme`.
- `Benchmark Snapshot` workflow now commits README benchmark updates.
- CI benchmark gating now enforces tighter zig/js slowdown limits and README benchmark sync.
- CI and release-prep now run tree hot-path microbench + regression checks.
- Release-native-prebuild workflow now runs only on published releases (plus manual dispatch) to avoid duplicate tag-triggered asset jobs.
- Hot-path predict retention guard for random-forest is relaxed to `0.55` in CI/release-prep to match observed stable baseline variance.
- CI now uses fast PR checks and main-only heavy lanes (native matrix, zig-tree smoke, benchmarks) to reduce PR cycle time while preserving release strictness.
- New CI `parity` job runs `parity:check` and enforces per-family sklearn drift thresholds.
- Native tree/forest ABI now supports multiclass labels end-to-end in Zig (`Uint16` labels, up to 256 classes) and is loaded via ABI version `2`.
- `HistGradientBoostingClassifier` / `HistGradientBoostingRegressor` now support `maxDepth`, `maxLeafNodes`, and early-stopping controls (`earlyStopping`, `nIterNoChange`, `validationFraction`, `tolerance`).
- sklearn fixtures and parity checks now include multi-seed drift baselines with fixture-defined threshold tables.

### Improved
- Zig tree codebase split into modules: `zig/src/tree/split.zig`, `zig/src/tree/fit.zig`, and `zig/src/tree/predict.zig`.
- RandomForest Zig fit uses dynamic atomic work scheduling across threads instead of static chunking.
- RandomForest Zig predict supports threaded row-chunk execution for larger inference batches.
- DecisionTree Zig predict includes a SIMD threshold-compare traversal path for larger row batches.
- Tree splitter threshold bin cap increased to `128` with exact-threshold fallback on small nodes.

## [0.1.6] - 2026-02-25

### Added
- CI Zig backend guard test (`test/zig-backend-guard.test.ts`) and enforced zig-tree smoke job gate (`BUN_SCIKIT_REQUIRE_ZIG_BACKEND=1`).

### Changed
- `Publish to npm` workflow now builds native artifacts in-job (Linux + Windows), assembles `prebuilt/*` from those fresh outputs, runs a consumer smoke test from `npm pack`, and only then publishes.
- Tree/forest backend path remains Zig-first with JS fallback, with stricter CI verification in Zig mode.

### Fixed
- Native kernel loading now tolerates prebuilt libraries that do not export random-forest symbols while still loading linear/logistic/tree symbols.

## [0.1.4] - 2026-02-23

### Added
- New sklearn-style APIs:
  - Baselines: `DummyClassifier`, `DummyRegressor`
  - Preprocessing: `MaxAbsScaler`, `Binarizer`, `LabelEncoder`, `Normalizer`
  - Feature selection: `VarianceThreshold`
  - Model selection: `RandomizedSearchCV`
  - Metrics: `balancedAccuracyScore`, `matthewsCorrcoef`, `brierScoreLoss`, `meanAbsolutePercentageError`, `explainedVarianceScore`
- Tree backend mode benchmarking (`js-fast` vs `zig-tree` vs `python-scikit-learn`) in CI benchmark snapshots.
- Dedicated tree backend control via `BUN_SCIKIT_TREE_BACKEND=zig`.

### Changed
- Optimized Zig decision-tree split kernel hot path using a binned splitter.
- Wired DecisionTree native fit/predict through runtime kernel loading with safe JS fallback.
- Extended Node-API addon to expose decision-tree native symbols.
- Added benchmark health guardrails for tree/forest and zig-vs-js backend slowdown limits.
- Updated README parity matrix and performance snapshot details.

## [0.1.3] - 2026-02-23

### Added
- Maintainer documentation baseline (`CONTRIBUTING`, `SECURITY`, `CODE_OF_CONDUCT`, `LICENSE`).
- `LogisticRegression` and `KNeighborsClassifier`.
- `DecisionTreeClassifier` and `RandomForestClassifier`.
- Classification metrics: `accuracyScore`, `precisionScore`, `recallScore`, `f1Score`.
- Heart dataset classification integration and model tests.
- Benchmark automation for Bun vs Python scikit-learn on `test_data/heart.csv` for regression, classification, and tree classification.
- CI benchmark workflows with snapshot history tracking and README benchmark sync/check tooling.
- API docs quality gates via Typedoc generation and exported-symbol coverage checks.
- Release checklist (`docs/release-checklist.md`).
- Consumer smoke test in CI to verify `bun add bun-scikit` works without trust-based install scripts.
- Benchmark health speedup floors to prevent regressions vs Python scikit-learn.

### Changed
- Bundled Linux/Windows prebuilt native binaries directly in npm package to avoid trust-required install hooks.

### Deprecated
- Dependency install-script bootstrap for downloading/building native artifacts at install time.

## [0.1.0] - 2026-02-22

### Added
- Initial `bun-scikit` package scaffold.
- `StandardScaler`.
- `LinearRegression` with `normal` and `gd` solvers.
- `trainTestSplit`.
- Regression metrics: `meanSquaredError`, `meanAbsoluteError`, `r2Score`.
- Unit and integration tests.
- Initial benchmark scripts.
