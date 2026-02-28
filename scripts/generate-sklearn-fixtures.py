#!/usr/bin/env python3
"""Generate sklearn snapshot fixtures for bun-scikit parity checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import AffinityPropagation, MeanShift, MiniBatchKMeans
from sklearn.decomposition import (
    FactorAnalysis,
    IncrementalPCA,
    KernelPCA,
    MiniBatchNMF,
    NMF,
)
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, GroupShuffleSplit, StratifiedGroupKFold, cross_val_predict
from sklearn.inspection import permutation_importance
from sklearn.multioutput import (
    ClassifierChain,
    MultiOutputClassifier,
    MultiOutputRegressor,
    RegressorChain,
)
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


def sort_rows_lexicographic(X: np.ndarray) -> np.ndarray:
    if X.size == 0:
        return X.copy()
    keys = tuple(X[:, col] for col in range(X.shape[1] - 1, -1, -1))
    order = np.lexsort(keys)
    return X[order]


def build_fixtures(seed: int, seeds: list[int]) -> dict:
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

    X_cluster_extra = np.array(
        [
            [0.0, 0.0],
            [0.1, -0.1],
            [0.2, 0.1],
            [5.0, 5.0],
            [5.1, 4.9],
            [4.9, 5.2],
        ],
        dtype=float,
    )
    mbk = MiniBatchKMeans(
        n_clusters=2,
        batch_size=3,
        max_iter=120,
        random_state=seed,
        n_init=5,
    ).fit(X_cluster_extra)
    mbk_centers = sort_rows_lexicographic(mbk.cluster_centers_)

    meanshift = MeanShift(bandwidth=1.2, max_iter=50, cluster_all=True).fit(X_cluster_extra)
    meanshift_centers = sort_rows_lexicographic(meanshift.cluster_centers_)

    affinity = AffinityPropagation(
        damping=0.7,
        max_iter=300,
        convergence_iter=30,
        random_state=seed,
    ).fit(X_cluster_extra)
    affinity_centers = sort_rows_lexicographic(affinity.cluster_centers_)

    X_ipca = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=float,
    )
    ipca = IncrementalPCA(n_components=2, batch_size=2)
    ipca.partial_fit(X_ipca[:2])
    ipca.partial_fit(X_ipca[2:])
    ipca_transform = ipca.transform(X_ipca)

    X_fa = np.array(
        [
            [1.0, 2.0, 0.0],
            [2.0, 3.0, 1.0],
            [3.0, 4.0, 1.0],
            [4.0, 5.0, 2.0],
            [5.0, 6.0, 2.0],
        ],
        dtype=float,
    )
    fa = FactorAnalysis(n_components=2, random_state=seed).fit(X_fa)
    fa_latent = fa.transform(X_fa)

    X_mbnmf = np.array(
        [
            [1.0, 0.5, 0.2],
            [0.8, 0.4, 0.1],
            [0.2, 0.9, 1.1],
            [0.1, 1.0, 1.2],
        ],
        dtype=float,
    )
    mbnmf = MiniBatchNMF(
        n_components=2,
        batch_size=2,
        max_iter=120,
        random_state=seed,
    )
    mbnmf_W = mbnmf.fit_transform(X_mbnmf)
    mbnmf_H = mbnmf.components_
    mbnmf_recon = mbnmf_W @ mbnmf_H

    X_multioutput = np.array(
        [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]],
        dtype=float,
    )
    Y_multioutput_clf = np.array(
        [
            [0, 0],
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 1],
            [1, 1],
        ],
        dtype=int,
    )
    Y_multioutput_reg = np.array(
        [
            [0.0, 1.0],
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
        ],
        dtype=float,
    )
    mo_clf = MultiOutputClassifier(DecisionTreeClassifier(max_depth=3, random_state=seed)).fit(
        X_multioutput,
        Y_multioutput_clf,
    )
    chain_clf = ClassifierChain(
        DecisionTreeClassifier(max_depth=3, random_state=seed),
        order=np.array([1, 0]),
    ).fit(X_multioutput, Y_multioutput_clf)
    mo_reg = MultiOutputRegressor(LinearRegression()).fit(X_multioutput, Y_multioutput_reg)
    chain_reg = RegressorChain(LinearRegression(), order=np.array([1, 0])).fit(
        X_multioutput,
        Y_multioutput_reg,
    )

    X_split = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.2, 0.1, 1.2],
            [0.4, 0.2, 1.1],
            [1.5, 1.4, 0.9],
            [1.6, 1.5, 1.0],
            [1.7, 1.6, 1.1],
            [3.0, 0.2, 0.8],
            [3.2, 0.3, 0.7],
            [3.4, 0.4, 0.9],
            [4.2, 1.8, 1.3],
            [4.3, 1.9, 1.4],
            [4.4, 2.0, 1.5],
        ],
        dtype=float,
    )
    y_split = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0], dtype=int)
    groups_split = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], dtype=int)

    gss = GroupShuffleSplit(n_splits=4, test_size=0.25, random_state=seed)
    gss_test_indices: list[list[int]] = []
    gss_test_group_ids: list[list[int]] = []
    gss_test_positive_rate: list[float] = []
    for _, test_idx in gss.split(X_split, y_split, groups_split):
        idx_list = test_idx.tolist()
        gss_test_indices.append(idx_list)
        gss_test_group_ids.append(sorted(np.unique(groups_split[test_idx]).tolist()))
        gss_test_positive_rate.append(float(np.mean(y_split[test_idx])))

    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    sgkf_test_indices: list[list[int]] = []
    sgkf_test_group_ids: list[list[int]] = []
    sgkf_test_positive_rate: list[float] = []
    for _, test_idx in sgkf.split(X_split, y_split, groups_split):
        idx_list = test_idx.tolist()
        sgkf_test_indices.append(idx_list)
        sgkf_test_group_ids.append(sorted(np.unique(groups_split[test_idx]).tolist()))
        sgkf_test_positive_rate.append(float(np.mean(y_split[test_idx])))

    X_perm = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [3.0, 1.0],
            [4.0, 0.0],
            [5.0, 1.0],
            [6.0, 0.0],
            [7.0, 1.0],
        ],
        dtype=float,
    )
    y_perm = np.array([0, 0, 0, 1, 1, 1, 1, 1], dtype=int)
    perm_estimator = LogisticRegression(max_iter=400, random_state=seed).fit(X_perm, y_perm)
    perm = permutation_importance(
        perm_estimator,
        X_perm,
        y_perm,
        scoring="accuracy",
        n_repeats=10,
        random_state=11,
    )

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
            "splitter_group_shuffle_rate_mse": 0.25,
            "splitter_stratified_group_rate_mse": 0.15,
            "inspection_permutation_mean_mse": 0.08,
            "inspection_permutation_rank_mismatch": 0.5,
            "minibatch_kmeans_center_mse": 0.5,
            "minibatch_kmeans_inertia_mse": 5.0,
            "meanshift_center_mse": 0.6,
            "affinity_center_mse": 1.2,
            "incremental_pca_transform_distance_mse": 1e-8,
            "factor_analysis_latent_distance_mse": 0.7,
            "minibatch_nmf_reconstruction_mse": 0.2,
            "multioutput_classifier_mismatch": 0.35,
            "classifier_chain_mismatch": 0.4,
            "multioutput_regressor_mse": 0.2,
            "regressor_chain_mse": 0.25,
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
        "additional_estimators": {
            "cluster": {
                "X": X_cluster_extra.tolist(),
                "minibatch_kmeans_centers": mbk_centers.tolist(),
                "minibatch_kmeans_inertia": float(mbk.inertia_),
                "meanshift_centers": meanshift_centers.tolist(),
                "affinity_centers": affinity_centers.tolist(),
            },
            "decomposition": {
                "incremental_pca_X": X_ipca.tolist(),
                "incremental_pca_transform": ipca_transform.tolist(),
                "factor_analysis_X": X_fa.tolist(),
                "factor_analysis_latent": fa_latent.tolist(),
                "minibatch_nmf_X": X_mbnmf.tolist(),
                "minibatch_nmf_reconstruction": mbnmf_recon.tolist(),
            },
            "multioutput": {
                "X": X_multioutput.tolist(),
                "Y_classifier": Y_multioutput_clf.tolist(),
                "Y_regressor": Y_multioutput_reg.tolist(),
                "multioutput_classifier_pred": mo_clf.predict(X_multioutput).tolist(),
                "classifier_chain_pred": chain_clf.predict(X_multioutput).tolist(),
                "multioutput_regressor_pred": mo_reg.predict(X_multioutput).tolist(),
                "regressor_chain_pred": chain_reg.predict(X_multioutput).tolist(),
            },
        },
        "splitters": {
            "X": X_split.tolist(),
            "y": y_split.tolist(),
            "groups": groups_split.tolist(),
            "group_shuffle_split": {
                "test_indices": gss_test_indices,
                "test_group_ids": gss_test_group_ids,
                "test_positive_rate": gss_test_positive_rate,
            },
            "stratified_group_kfold": {
                "test_indices": sgkf_test_indices,
                "test_group_ids": sgkf_test_group_ids,
                "test_positive_rate": sgkf_test_positive_rate,
            },
        },
        "inspection": {
            "permutation_importance": {
                "X": X_perm.tolist(),
                "y": y_perm.tolist(),
                "importances_mean": perm.importances_mean.tolist(),
                "importances_std": perm.importances_std.tolist(),
                "importances": perm.importances.tolist(),
            }
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
