# Parity Scope (Active)

## Core Problem
Deliver practical scikit-learn parity for Bun/TypeScript without stalling releases on full long-tail coverage.

## Success Criteria
- Users can build end-to-end classic ML workflows (preprocess, split, train, tune, evaluate) fully in `bun-scikit`.
- Each new parity area ships with tests, docs, and deterministic behavior.
- Releases continue on small increments instead of a single "big bang" parity attempt.

## In Scope (Current Wave)
- Add foundational clustering parity: `KMeans`, `DBSCAN`, `AgglomerativeClustering`.
- Add foundational decomposition parity: `PCA`.
- Add classification calibration/meta-estimator baseline parity: `CalibratedClassifierCV`, `VotingClassifier`, `StackingClassifier`.
- Keep API style aligned with existing estimator conventions (`fit`/`predict`/`transform`, learned attrs).

## Explicitly Out of Scope (Current Wave)
- Full scikit module parity in one release.
- Advanced clustering families (`DBSCAN`, agglomerative, spectral).
- Advanced decomposition (`TruncatedSVD`, `FastICA`, kernel methods).
- Calibration/meta-estimators and manifold modules.

## Non-Negotiables
- New APIs must have unit tests.
- Existing tests and typecheck must stay green.
- Scope additions require explicit tracking in this file.
