# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Reusable `Release Prep` workflow (`.github/workflows/release-prep.yml`) that gates release pipelines with tests, typecheck, Zig guard checks, benchmark checks, README benchmark sync checks, and npm-pack smoke validation.
- Zig backend smoke example for users: `examples/zig-backend-smoke.ts`.

### Changed
- README install docs now include a post-install Zig backend smoke check for `DecisionTreeClassifier` and `RandomForestClassifier`.
- README benchmark section is now marker-driven and auto-generated from `bench/results/heart-ci-latest.json` via `scripts/sync-benchmark-readme.ts`.
- `bench:snapshot` now also runs `bench:sync-readme`.
- `Benchmark Snapshot` workflow now commits README benchmark updates.
- CI benchmark gating now enforces tighter zig/js slowdown limits and README benchmark sync.

### Improved
- Zig tree predict path uses row-pointer traversal and compact `u32` node indices for better cache behavior.
- Tree splitter threshold bin cap increased from 24 to 32 for improved split quality.

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
