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
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def build_fixtures(seed: int) -> dict:
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

    return {
        "metadata": {
            "description": "Snapshot fixtures for parity checks",
            "seed": seed,
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    fixture = build_fixtures(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
