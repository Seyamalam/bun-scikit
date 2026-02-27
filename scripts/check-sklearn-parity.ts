import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import {
  CalibratedClassifierCV,
  DecisionTreeClassifier,
  GaussianNB,
  HistGradientBoostingClassifier,
  HistGradientBoostingRegressor,
  KernelPCA,
  KFold,
  KNeighborsClassifier,
  LogisticRegression,
  NMF,
  Pipeline,
  RandomForestClassifier,
  StandardScaler,
  VotingClassifier,
  crossValPredict,
} from "../src";

type Matrix = number[][];
type Vector = number[];

function meanAbsDiffMatrix(a: Matrix, b: Matrix): number {
  let total = 0;
  let count = 0;
  for (let i = 0; i < a.length; i += 1) {
    for (let j = 0; j < a[i].length; j += 1) {
      total += Math.abs(a[i][j] - b[i][j]);
      count += 1;
    }
  }
  return total / Math.max(1, count);
}

function meanSquaredErrorVector(a: Vector, b: Vector): number {
  let total = 0;
  for (let i = 0; i < a.length; i += 1) {
    const diff = a[i] - b[i];
    total += diff * diff;
  }
  return total / Math.max(1, a.length);
}

function meanSquaredErrorMatrix(a: Matrix, b: Matrix): number {
  let total = 0;
  let count = 0;
  for (let i = 0; i < a.length; i += 1) {
    for (let j = 0; j < a[i].length; j += 1) {
      const diff = a[i][j] - b[i][j];
      total += diff * diff;
      count += 1;
    }
  }
  return total / Math.max(1, count);
}

function mismatchRate(a: Vector, b: Vector): number {
  let mismatches = 0;
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) {
      mismatches += 1;
    }
  }
  return mismatches / Math.max(1, a.length);
}

function pairwiseDistanceMatrix(X: Matrix): Matrix {
  const out: Matrix = Array.from({ length: X.length }, () => new Array<number>(X.length).fill(0));
  for (let i = 0; i < X.length; i += 1) {
    for (let j = i + 1; j < X.length; j += 1) {
      let sum = 0;
      for (let k = 0; k < X[i].length; k += 1) {
        const d = X[i][k] - X[j][k];
        sum += d * d;
      }
      const dist = Math.sqrt(sum);
      out[i][j] = dist;
      out[j][i] = dist;
    }
  }
  return out;
}

function threshold(name: string, fallback: number): number {
  const raw = process.env[name];
  if (!raw) {
    return fallback;
  }
  const parsed = Number(raw);
  if (!Number.isFinite(parsed)) {
    throw new Error(`Invalid numeric threshold for ${name}: ${raw}`);
  }
  return parsed;
}

const fixture = JSON.parse(readFileSync(resolve("test/fixtures/sklearn-snapshots.json"), "utf-8"));
const fixtureThresholds: Record<string, number> = fixture.thresholds ?? {};

const metrics: Record<string, number> = {};

{
  const model = new GaussianNB().fit(fixture.multiclass.X, fixture.multiclass.y);
  metrics.gnb_proba_mad = meanAbsDiffMatrix(
    model.predictProba(fixture.multiclass.probe),
    fixture.multiclass.gaussian_nb_proba,
  );
}

{
  const model = new VotingClassifier(
    [
      ["gnb", () => new GaussianNB()],
      ["knn", () => new KNeighborsClassifier({ nNeighbors: 3 })],
    ],
    { voting: "soft" },
  ).fit(fixture.multiclass.X, fixture.multiclass.y);
  metrics.voting_soft_proba_mad = meanAbsDiffMatrix(
    model.predictProba(fixture.multiclass.probe),
    fixture.multiclass.voting_soft_proba,
  );
}

{
  const model = new CalibratedClassifierCV(() => new GaussianNB(), {
    cv: 3,
    method: "sigmoid",
    ensemble: false,
    randomState: 42,
  }).fit(fixture.multiclass.X, fixture.multiclass.y);
  metrics.calibrated_proba_mad = meanAbsDiffMatrix(
    model.predictProba(fixture.multiclass.probe),
    fixture.multiclass.calibrated_sigmoid_proba,
  );
}

{
  const previousTreeBackend = process.env.BUN_SCIKIT_TREE_BACKEND;
  process.env.BUN_SCIKIT_TREE_BACKEND = "js";
  try {
    const dt = new DecisionTreeClassifier({ maxDepth: 4, randomState: 42 }).fit(
      fixture.multiclass.X,
      fixture.multiclass.y,
    );
    metrics.decision_tree_mismatch = mismatchRate(
      dt.predict(fixture.multiclass.X),
      fixture.multiclass.decision_tree_pred,
    );

    const rf = new RandomForestClassifier({
      nEstimators: 40,
      maxDepth: 4,
      randomState: 42,
    }).fit(fixture.multiclass.X, fixture.multiclass.y);
    metrics.random_forest_mismatch = mismatchRate(
      rf.predict(fixture.multiclass.X),
      fixture.multiclass.random_forest_pred,
    );
  } finally {
    if (previousTreeBackend === undefined) {
      delete process.env.BUN_SCIKIT_TREE_BACKEND;
    } else {
      process.env.BUN_SCIKIT_TREE_BACKEND = previousTreeBackend;
    }
  }
}

{
  const model = new HistGradientBoostingClassifier({
    maxIter: 120,
    learningRate: 0.08,
    maxBins: 16,
    randomState: 42,
  }).fit(fixture.hist_gradient_boosting.X_binary, fixture.hist_gradient_boosting.y_binary);
  metrics.hist_gb_classifier_probe_mad = meanAbsDiffMatrix(
    model.predictProba(fixture.hist_gradient_boosting.probe_binary),
    fixture.hist_gradient_boosting.classifier_probe_proba,
  );
  metrics.hist_gb_classifier_mismatch = mismatchRate(
    model.predict(fixture.hist_gradient_boosting.X_binary),
    fixture.hist_gradient_boosting.classifier_train_pred,
  );
}

{
  const model = new HistGradientBoostingRegressor({
    maxIter: 150,
    learningRate: 0.08,
    maxBins: 16,
    randomState: 42,
  }).fit(fixture.hist_gradient_boosting.X_reg, fixture.hist_gradient_boosting.y_reg);
  metrics.hist_gb_regressor_probe_mse = meanSquaredErrorVector(
    model.predict(fixture.hist_gradient_boosting.probe_reg),
    fixture.hist_gradient_boosting.regressor_probe_pred,
  );
  metrics.hist_gb_regressor_train_mse = meanSquaredErrorVector(
    model.predict(fixture.hist_gradient_boosting.X_reg),
    fixture.hist_gradient_boosting.regressor_train_pred,
  );
}

{
  const model = new NMF({
    nComponents: 2,
    maxIter: 500,
    tolerance: 1e-6,
    randomState: 42,
  });
  const W = model.fitTransform(fixture.nmf.X);
  const recon = model.inverseTransform(W);
  metrics.nmf_reconstruction_mse = meanSquaredErrorMatrix(recon, fixture.nmf.reconstruction);
}

{
  const model = new KernelPCA({
    nComponents: 2,
    kernel: "rbf",
    gamma: 0.5,
  }).fit(fixture.kernel_pca.X);
  const train = model.transform(fixture.kernel_pca.X);
  const probe = model.transform(fixture.kernel_pca.probe);
  metrics.kernel_pca_train_distance_mse = meanSquaredErrorMatrix(
    pairwiseDistanceMatrix(train),
    pairwiseDistanceMatrix(fixture.kernel_pca.train_transform),
  );
  metrics.kernel_pca_probe_distance_mse = meanSquaredErrorMatrix(
    pairwiseDistanceMatrix(probe),
    pairwiseDistanceMatrix(fixture.kernel_pca.probe_transform),
  );
}

{
  const section = fixture.pipeline_logistic_regression;
  const model = new Pipeline([
    ["scale", new StandardScaler()],
    [
      "clf",
      new LogisticRegression({
        maxIter: 400,
        learningRate: 0.1,
        tolerance: 1e-6,
      }),
    ],
  ]).fit(section.X, section.y);

  metrics.pipeline_logreg_probe_proba_mad = meanAbsDiffMatrix(
    model.predictProba(section.probe),
    section.probe_proba,
  );
  metrics.pipeline_logreg_train_mismatch = mismatchRate(
    model.predict(section.X),
    section.train_pred,
  );

  const oofPred = crossValPredict(
    () =>
      new Pipeline([
        ["scale", new StandardScaler()],
        [
          "clf",
          new LogisticRegression({
            maxIter: 400,
            learningRate: 0.1,
            tolerance: 1e-6,
          }),
        ],
      ]),
    section.X,
    section.y,
    { cv: new KFold({ nSplits: 4, shuffle: false }) },
  ) as Vector;
  metrics.pipeline_cv_predict_mismatch = mismatchRate(oofPred, section.cv_predict_kfold4);
}

{
  const seeds: number[] = fixture.multi_seed.seeds;
  const previousTreeBackend = process.env.BUN_SCIKIT_TREE_BACKEND;
  process.env.BUN_SCIKIT_TREE_BACKEND = "js";
  try {
    let dtMismatchTotal = 0;
    let rfMismatchTotal = 0;
    for (let i = 0; i < seeds.length; i += 1) {
      const seed = seeds[i];
      const dt = new DecisionTreeClassifier({ maxDepth: 4, randomState: seed }).fit(
        fixture.multiclass.X,
        fixture.multiclass.y,
      );
      dtMismatchTotal += mismatchRate(dt.predict(fixture.multiclass.X), fixture.multi_seed.decision_tree_pred[i]);

      const rf = new RandomForestClassifier({
        nEstimators: 40,
        maxDepth: 4,
        randomState: seed,
      }).fit(fixture.multiclass.X, fixture.multiclass.y);
      rfMismatchTotal += mismatchRate(rf.predict(fixture.multiclass.X), fixture.multi_seed.random_forest_pred[i]);
    }
    metrics.multi_seed_decision_tree_mismatch_avg = dtMismatchTotal / seeds.length;
    metrics.multi_seed_random_forest_mismatch_avg = rfMismatchTotal / seeds.length;
  } finally {
    if (previousTreeBackend === undefined) {
      delete process.env.BUN_SCIKIT_TREE_BACKEND;
    } else {
      process.env.BUN_SCIKIT_TREE_BACKEND = previousTreeBackend;
    }
  }
}

const limits: Record<string, number> = {
  gnb_proba_mad: threshold("PARITY_MAX_GNB_PROBA_MAD", fixtureThresholds.gnb_proba_mad ?? 0.15),
  voting_soft_proba_mad: threshold(
    "PARITY_MAX_VOTING_SOFT_PROBA_MAD",
    fixtureThresholds.voting_soft_proba_mad ?? 0.2,
  ),
  calibrated_proba_mad: threshold(
    "PARITY_MAX_CALIBRATED_PROBA_MAD",
    fixtureThresholds.calibrated_proba_mad ?? 0.25,
  ),
  decision_tree_mismatch: threshold(
    "PARITY_MAX_DECISION_TREE_MISMATCH",
    fixtureThresholds.decision_tree_mismatch ?? 0.05,
  ),
  random_forest_mismatch: threshold(
    "PARITY_MAX_RANDOM_FOREST_MISMATCH",
    fixtureThresholds.random_forest_mismatch ?? 0.1,
  ),
  hist_gb_classifier_probe_mad: threshold(
    "PARITY_MAX_HIST_GB_CLASSIFIER_PROBE_MAD",
    fixtureThresholds.hist_gb_classifier_probe_mad ?? 0.25,
  ),
  hist_gb_classifier_mismatch: threshold(
    "PARITY_MAX_HIST_GB_CLASSIFIER_MISMATCH",
    fixtureThresholds.hist_gb_classifier_mismatch ?? 0.1,
  ),
  hist_gb_regressor_probe_mse: threshold(
    "PARITY_MAX_HIST_GB_REGRESSOR_PROBE_MSE",
    fixtureThresholds.hist_gb_regressor_probe_mse ?? 50,
  ),
  hist_gb_regressor_train_mse: threshold(
    "PARITY_MAX_HIST_GB_REGRESSOR_TRAIN_MSE",
    fixtureThresholds.hist_gb_regressor_train_mse ?? 50,
  ),
  nmf_reconstruction_mse: threshold(
    "PARITY_MAX_NMF_RECONSTRUCTION_MSE",
    fixtureThresholds.nmf_reconstruction_mse ?? 0.1,
  ),
  kernel_pca_train_distance_mse: threshold(
    "PARITY_MAX_KERNEL_PCA_TRAIN_DISTANCE_MSE",
    fixtureThresholds.kernel_pca_train_distance_mse ?? 0.2,
  ),
  kernel_pca_probe_distance_mse: threshold(
    "PARITY_MAX_KERNEL_PCA_PROBE_DISTANCE_MSE",
    fixtureThresholds.kernel_pca_probe_distance_mse ?? 0.2,
  ),
  multi_seed_decision_tree_mismatch_avg: threshold(
    "PARITY_MAX_MULTI_SEED_DECISION_TREE_MISMATCH_AVG",
    fixtureThresholds.multi_seed_decision_tree_mismatch_avg ?? 0.08,
  ),
  multi_seed_random_forest_mismatch_avg: threshold(
    "PARITY_MAX_MULTI_SEED_RANDOM_FOREST_MISMATCH_AVG",
    fixtureThresholds.multi_seed_random_forest_mismatch_avg ?? 0.12,
  ),
  pipeline_logreg_probe_proba_mad: threshold(
    "PARITY_MAX_PIPELINE_LOGREG_PROBE_PROBA_MAD",
    fixtureThresholds.pipeline_logreg_probe_proba_mad ?? 0.15,
  ),
  pipeline_logreg_train_mismatch: threshold(
    "PARITY_MAX_PIPELINE_LOGREG_TRAIN_MISMATCH",
    fixtureThresholds.pipeline_logreg_train_mismatch ?? 0.12,
  ),
  pipeline_cv_predict_mismatch: threshold(
    "PARITY_MAX_PIPELINE_CV_PREDICT_MISMATCH",
    fixtureThresholds.pipeline_cv_predict_mismatch ?? 0.2,
  ),
};

let failed = false;
for (const [name, value] of Object.entries(metrics)) {
  const limit = limits[name];
  const pass = value <= limit;
  const status = pass ? "PASS" : "FAIL";
  console.log(`${status} ${name}: value=${value} limit=${limit}`);
  if (!pass) {
    failed = true;
  }
}

if (failed) {
  console.error("sklearn parity check failed.");
  process.exit(1);
}

console.log("sklearn parity check passed.");
