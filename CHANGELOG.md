# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Maintainer documentation baseline (`CONTRIBUTING`, `SECURITY`, `CODE_OF_CONDUCT`, `LICENSE`).
- `LogisticRegression` and `KNeighborsClassifier`.
- `DecisionTreeClassifier` and `RandomForestClassifier`.
- Classification metrics: `accuracyScore`, `precisionScore`, `recallScore`, `f1Score`.
- Heart dataset classification integration and model tests.
- Benchmark automation for Bun vs Python scikit-learn on `test_data/heart.csv` for regression, classification, and tree classification.
- CI benchmark workflows with snapshot history tracking and README benchmark sync/check tooling.
- API docs quality gates via Typedoc generation and exported-symbol coverage checks.

## [0.1.0] - 2026-02-22

### Added
- Initial `bun-scikit` package scaffold.
- `StandardScaler`.
- `LinearRegression` with `normal` and `gd` solvers.
- `trainTestSplit`.
- Regression metrics: `meanSquaredError`, `meanAbsoluteError`, `r2Score`.
- Unit and integration tests.
- Initial benchmark scripts.
