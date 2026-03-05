# Parity Scope (Active)

## Core Problem
Deliver practical scikit-learn parity for Bun/TypeScript without stalling releases on full long-tail coverage.

## Success Criteria
- Users can build end-to-end classic ML workflows (preprocess, split, train, tune, evaluate) fully in `bun-scikit`.
- Each new parity area ships with tests, docs, and deterministic behavior.
- Releases continue on small increments instead of a single "big bang" parity attempt.

## In Scope (Current Wave)
- Maintain shipped clustering parity: `KMeans`, `DBSCAN`, `AgglomerativeClustering`.
- Maintain shipped decomposition parity: `PCA`, `TruncatedSVD`, `FastICA`, `NMF`, `KernelPCA`.
- Maintain shipped calibration/meta-estimator parity: `CalibratedClassifierCV`, `VotingClassifier`, `VotingRegressor`, `StackingClassifier`, `StackingRegressor`, `BaggingClassifier`.
- Maintain shipped boosting parity baseline: `AdaBoostClassifier`, `GradientBoostingClassifier`, `GradientBoostingRegressor`, `HistGradientBoostingClassifier`, `HistGradientBoostingRegressor`.
- Maintain multiclass support baseline across linear, probabilistic, neighbor, tree/forest, and meta-ensemble classifiers.
- Maintain multiclass native Zig tree/forest backend parity (no binary-only native restriction).
- Expand model-selection splitter parity with classical CV splitters (`ShuffleSplit`, leave-one/leave-p-out variants, `PredefinedSplit`, `TimeSeriesSplit`).
- Keep API style aligned with existing estimator conventions (`fit`/`predict`/`transform`, learned attrs).
- Maintain estimator introspection baseline via learned feature-importance attributes for tree/forest/boosting families.
- Maintain parity guardrails with reproducible fixture generation and multi-seed thresholded drift checks.
- Maintain machine-readable parity matrix coverage (`docs/parity-matrix.json`) enforced in CI/release gates.

## Explicitly Out of Scope (Current Wave)
- Full scikit module parity in one release.
- Advanced clustering families (spectral/OPTICS/HDBSCAN-like parity).
- Advanced decomposition beyond current baseline (sparse coding, incremental decomposition).
- Boosting/manifold/inspection module breadth (extra-trees, XGBoost-style APIs, partial-dependence/SHAP).

## Non-Negotiables
- New APIs must have unit tests.
- Existing tests and typecheck must stay green.
- Scope additions require explicit tracking in this file.
