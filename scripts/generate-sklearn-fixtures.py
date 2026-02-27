#!/usr/bin/env python3
"""Generate sklearn snapshot fixtures for bun-scikit parity checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import KernelPCA, NMF
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


def parse_seed_list(raw: str) -> list[int]:
    out: list[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        out.append(int(chunk))
    if not out:
        raise ValueError("At least one seed must be provided.")
    return out


def build_fixtures(seed: int, seeds: list[int]) -> dict:
    rng = np.random.default_rng(seed)

    X_multi = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.2],
            [0.2, -0.1],
            [2.0, 2.1],
            [2.2, 1.9],
            [1.8, 2.0],
            [4.0, 4.2],
            [3.9, 4.1],
            [4.2, 3.8],
        ],
        dtype=float,
    )
    y_multi = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=int)
    probe_multi = np.array([[0.15, 0.05], [2.05, 2.0], [4.05, 4.0]], dtype=float)

    gnb = GaussianNB().fit(X_multi, y_multi)
    gnb_proba = gnb.predict_proba(probe_multi)

    voting = VotingClassifier(
        estimators=[("gnb", GaussianNB()), ("knn", KNeighborsClassifier(n_neighbors=3))],
        voting="soft",
    ).fit(X_multi, y_multi)
    voting_proba = voting.predict_proba(probe_multi)

    cal = CalibratedClassifierCV(GaussianNB(), method="sigmoid", cv=3, ensemble=False)
    cal.fit(X_multi, y_multi)
    cal_proba = cal.predict_proba(probe_multi)

    dt = DecisionTreeClassifier(max_depth=4, random_state=seed).fit(X_multi, y_multi)
    dt_pred = dt.predict(X_multi)

    rf = RandomForestClassifier(
        n_estimators=40,
        max_depth=4,
        random_state=seed,
    ).fit(X_multi, y_multi)
    rf_pred = rf.predict(X_multi)
    rf_probe_proba = rf.predict_proba(probe_multi)

    X_binary = np.array([[-4], [-3], [-2], [-1], [1], [2], [3], [4], [5], [6]], dtype=float)
    y_binary = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=int)
    probe_binary = np.array([[-3], [5]], dtype=float)

    pipeline_logreg = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=400, random_state=seed)),
        ]
    ).fit(X_binary, y_binary)
    pipeline_logreg_probe_proba = pipeline_logreg.predict_proba(probe_binary)
    pipeline_logreg_train_pred = pipeline_logreg.predict(X_binary)
    pipeline_cv_pred = cross_val_predict(
        Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=400, random_state=seed)),
            ]
        ),
        X_binary,
        y_binary,
        cv=KFold(n_splits=4, shuffle=False),
        method="predict",
    )

    X_composition = np.array(
        [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 10.0, 300.0],
            [4.0, 30.0, 400.0],
        ],
        dtype=float,
    )

    pipeline_scaler_transform = Pipeline(
        steps=[("scaler", StandardScaler())]
    ).fit_transform(X_composition)
    column_transformer_transform = ColumnTransformer(
        transformers=[
            ("scale_col0", MinMaxScaler(), [0]),
            ("encode_col1", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), [1]),
        ],
        remainder="passthrough",
    ).fit_transform(X_composition)

    hgb_clf = HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.08,
        max_bins=16,
        random_state=seed,
    ).fit(X_binary, y_binary)
    hgb_clf_proba = hgb_clf.predict_proba(probe_binary)
    hgb_clf_pred = hgb_clf.predict(X_binary)

    X_reg = np.arange(0, 10, dtype=float).reshape(-1, 1)
    y_reg = np.array([0, 1, 4, 9, 16, 25, 36, 49, 64, 81], dtype=float)
    probe_reg = np.array([[1.5], [3.5], [7.5]], dtype=float)

    hgb_reg = HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.08,
        max_bins=16,
        random_state=seed,
    ).fit(X_reg, y_reg)
    hgb_reg_pred = hgb_reg.predict(probe_reg)
    hgb_reg_train_pred = hgb_reg.predict(X_reg)

    X_nmf = np.array(
        [
            [1.0, 0.5, 0.2],
            [0.9, 0.4, 0.3],
            [0.2, 0.8, 1.0],
            [0.1, 0.9, 1.1],
        ],
        dtype=float,
    )
    nmf = NMF(n_components=2, init="random", random_state=seed, max_iter=500, tol=1e-6)
    W = nmf.fit_transform(X_nmf)
    H = nmf.components_
    nmf_recon = W @ H

    X_kpca = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.2],
            [0.2, -0.1],
            [1.0, 1.1],
            [1.2, 0.9],
            [0.8, 1.2],
        ],
        dtype=float,
    )
    kpca = KernelPCA(n_components=2, kernel="rbf", gamma=0.5)
    kpca_train = kpca.fit_transform(X_kpca)
    kpca_probe = kpca.transform(np.array([[0.1, 0.1], [1.1, 1.0]], dtype=float))

    multi_seed_dt_preds: list[list[int]] = []
    multi_seed_rf_preds: list[list[int]] = []
    for current_seed in seeds:
        seeded_dt = DecisionTreeClassifier(max_depth=4, random_state=current_seed).fit(X_multi, y_multi)
        multi_seed_dt_preds.append(seeded_dt.predict(X_multi).tolist())
        seeded_rf = RandomForestClassifier(
            n_estimators=40,
            max_depth=4,
            random_state=current_seed,
        ).fit(X_multi, y_multi)
        multi_seed_rf_preds.append(seeded_rf.predict(X_multi).tolist())

    return {
        "metadata": {
            "description": "Snapshot fixtures for parity checks",
            "seed": seed,
            "seeds": seeds,
        },
        "thresholds": {
            "gnb_proba_mad": 0.15,
            "voting_soft_proba_mad": 0.2,
            "calibrated_proba_mad": 0.25,
            "decision_tree_mismatch": 0.05,
            "random_forest_mismatch": 0.1,
            "hist_gb_classifier_probe_mad": 0.25,
            "hist_gb_classifier_mismatch": 0.1,
            "hist_gb_regressor_probe_mse": 50.0,
            "hist_gb_regressor_train_mse": 50.0,
            "nmf_reconstruction_mse": 0.1,
            "kernel_pca_train_distance_mse": 0.2,
            "kernel_pca_probe_distance_mse": 0.2,
            "multi_seed_decision_tree_mismatch_avg": 0.08,
            "multi_seed_random_forest_mismatch_avg": 0.12,
            "pipeline_logreg_probe_proba_mad": 0.15,
            "pipeline_logreg_train_mismatch": 0.12,
            "pipeline_cv_predict_mismatch": 0.2,
            "composition_pipeline_transform_mse": 1e-10,
            "composition_column_transformer_mse": 1e-10,
        },
        "multiclass": {
            "X": X_multi.tolist(),
            "y": y_multi.tolist(),
            "probe": probe_multi.tolist(),
            "gaussian_nb_proba": gnb_proba.tolist(),
            "voting_soft_proba": voting_proba.tolist(),
            "calibrated_sigmoid_proba": cal_proba.tolist(),
            "decision_tree_pred": dt_pred.tolist(),
            "random_forest_pred": rf_pred.tolist(),
            "random_forest_probe_proba": rf_probe_proba.tolist(),
        },
        "multi_seed": {
            "seeds": seeds,
            "decision_tree_pred": multi_seed_dt_preds,
            "random_forest_pred": multi_seed_rf_preds,
        },
        "pipeline_logistic_regression": {
            "X": X_binary.tolist(),
            "y": y_binary.tolist(),
            "probe": probe_binary.tolist(),
            "probe_proba": pipeline_logreg_probe_proba.tolist(),
            "train_pred": pipeline_logreg_train_pred.tolist(),
            "cv_predict_kfold4": pipeline_cv_pred.tolist(),
        },
        "composition": {
            "X": X_composition.tolist(),
            "pipeline_scaler_transform": pipeline_scaler_transform.tolist(),
            "column_transformer_transform": column_transformer_transform.tolist(),
        },
        "hist_gradient_boosting": {
            "X_binary": X_binary.tolist(),
            "y_binary": y_binary.tolist(),
            "probe_binary": probe_binary.tolist(),
            "classifier_probe_proba": hgb_clf_proba.tolist(),
            "classifier_train_pred": hgb_clf_pred.tolist(),
            "X_reg": X_reg.tolist(),
            "y_reg": y_reg.tolist(),
            "probe_reg": probe_reg.tolist(),
            "regressor_probe_pred": hgb_reg_pred.tolist(),
            "regressor_train_pred": hgb_reg_train_pred.tolist(),
        },
        "nmf": {
            "X": X_nmf.tolist(),
            "W": W.tolist(),
            "H": H.tolist(),
            "reconstruction": nmf_recon.tolist(),
        },
        "kernel_pca": {
            "X": X_kpca.tolist(),
            "train_transform": kpca_train.tolist(),
            "probe": [[0.1, 0.1], [1.1, 1.0]],
            "probe_transform": kpca_probe.tolist(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sklearn parity fixtures")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test/fixtures/sklearn-snapshots.json"),
        help="Path for generated fixture JSON",
    )
    parser.add_argument("--seed", type=int, default=42, help="Primary random seed")
    parser.add_argument(
        "--seeds",
        type=str,
        default="13,42,77",
        help="Comma-separated seeds for multi-seed parity fixtures",
    )
    args = parser.parse_args()

    seeds = parse_seed_list(args.seeds)
    fixture = build_fixtures(args.seed, seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
