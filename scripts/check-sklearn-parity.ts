import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import {
  AffinityPropagation,
  CalibratedClassifierCV,
  ClassifierChain,
  ColumnTransformer,
  DecisionTreeClassifier,
  FactorAnalysis,
  GaussianNB,
  HistGradientBoostingClassifier,
  HistGradientBoostingRegressor,
  IncrementalPCA,
  KernelPCA,
  KFold,
  KNeighborsClassifier,
  LinearRegression,
  LogisticRegression,
  MeanShift,
  MinMaxScaler,
  MiniBatchKMeans,
  MiniBatchNMF,
  MultiOutputClassifier,
  MultiOutputRegressor,
  NMF,
  OneHotEncoder,
  Pipeline,
  GroupShuffleSplit,
  RandomForestClassifier,
  RegressorChain,
  StandardScaler,
  StratifiedGroupKFold,
  VotingClassifier,
  crossValPredict,
  permutationImportance,
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

function meanSquaredErrorMatrixWithShapePenalty(a: Matrix, b: Matrix): number {
  const rows = Math.min(a.length, b.length);
  if (rows === 0) {
    const diff = a.length - b.length;
    return diff * diff;
  }
  const cols = Math.min(a[0].length, b[0].length);
  if (cols === 0) {
    const diff = a[0].length - b[0].length;
    return diff * diff;
  }
  let total = 0;
  let count = 0;
  for (let i = 0; i < rows; i += 1) {
    for (let j = 0; j < cols; j += 1) {
      const d = a[i][j] - b[i][j];
      total += d * d;
      count += 1;
    }
  }
  const rowDiff = a.length - b.length;
  const colDiff = a[0].length - b[0].length;
  total += rowDiff * rowDiff + colDiff * colDiff;
  count += 1;
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

function mismatchRateMatrix(a: Matrix, b: Matrix): number {
  let mismatches = 0;
  let count = 0;
  for (let i = 0; i < a.length; i += 1) {
    for (let j = 0; j < a[i].length; j += 1) {
      if (a[i][j] !== b[i][j]) {
        mismatches += 1;
      }
      count += 1;
    }
  }
  return mismatches / Math.max(1, count);
}

function sortRowsLexicographic(X: Matrix): Matrix {
  return X
    .map((row) => row.slice())
    .sort((a, b) => {
      const len = Math.min(a.length, b.length);
      for (let i = 0; i < len; i += 1) {
        if (a[i] !== b[i]) {
          return a[i] - b[i];
        }
      }
      return a.length - b.length;
    });
}

function argsortDesc(values: number[]): number[] {
  return Array.from({ length: values.length }, (_, i) => i).sort((a, b) => {
    if (values[b] !== values[a]) {
      return values[b] - values[a];
    }
    return a - b;
  });
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
  const section = fixture.composition;
  const pipeline = new Pipeline([["scaler", new StandardScaler()]]);
  const pipelineOut = pipeline.fitTransform(section.X);
  metrics.composition_pipeline_transform_mse = meanSquaredErrorMatrix(
    pipelineOut,
    section.pipeline_scaler_transform,
  );

  const ct = new ColumnTransformer(
    [
      ["scale_col0", new MinMaxScaler(), [0]],
      ["encode_col1", new OneHotEncoder(), [1]],
    ],
    { remainder: "passthrough" },
  );
  const ctOut = ct.fitTransform(section.X);
  metrics.composition_column_transformer_mse = meanSquaredErrorMatrix(
    ctOut,
    section.column_transformer_transform,
  );
}

{
  const section = fixture.splitters;
  const gssSplits = new GroupShuffleSplit({
    nSplits: 4,
    testSize: 0.25,
    randomState: 42,
  }).split(section.X, section.y, section.groups);
  const gssRates = gssSplits.map((split) => {
    let positives = 0;
    for (let i = 0; i < split.testIndices.length; i += 1) {
      positives += section.y[split.testIndices[i]];
    }
    return positives / Math.max(1, split.testIndices.length);
  });
  const expectedGssRates = [...section.group_shuffle_split.test_positive_rate];
  gssRates.sort((a, b) => a - b);
  expectedGssRates.sort((a, b) => a - b);
  let gssRateMse = 0;
  for (let i = 0; i < gssRates.length; i += 1) {
    const d = gssRates[i] - expectedGssRates[i];
    gssRateMse += d * d;
  }
  metrics.splitter_group_shuffle_rate_mse = gssRateMse / Math.max(1, gssRates.length);

  const sgkfSplits = new StratifiedGroupKFold({
    nSplits: 3,
    shuffle: true,
    randomState: 42,
  }).split(section.X, section.y, section.groups);
  const sgkfRates = sgkfSplits.map((split) => {
    let positives = 0;
    for (let i = 0; i < split.testIndices.length; i += 1) {
      positives += section.y[split.testIndices[i]];
    }
    return positives / Math.max(1, split.testIndices.length);
  });
  const expectedSgkfRates = [...section.stratified_group_kfold.test_positive_rate];
  sgkfRates.sort((a, b) => a - b);
  expectedSgkfRates.sort((a, b) => a - b);
  let sgkfRateMse = 0;
  for (let i = 0; i < sgkfRates.length; i += 1) {
    const d = sgkfRates[i] - expectedSgkfRates[i];
    sgkfRateMse += d * d;
  }
  metrics.splitter_stratified_group_rate_mse = sgkfRateMse / Math.max(1, sgkfRates.length);
}

{
  const section = fixture.inspection.permutation_importance;
  const estimator = new LogisticRegression({
    maxIter: 400,
    learningRate: 0.2,
    tolerance: 1e-6,
  }).fit(section.X, section.y);
  const result = permutationImportance(estimator, section.X, section.y, {
    scoring: "accuracy",
    nRepeats: 10,
    randomState: 11,
  });
  metrics.inspection_permutation_mean_mse = meanSquaredErrorVector(
    result.importancesMean,
    section.importances_mean,
  );
  const predictedRank = argsortDesc(result.importancesMean);
  const expectedRank = argsortDesc(section.importances_mean);
  metrics.inspection_permutation_rank_mismatch = mismatchRate(predictedRank, expectedRank);
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

{
  const section = fixture.additional_estimators.cluster;
  const mbk = new MiniBatchKMeans({
    nClusters: 2,
    batchSize: 3,
    maxIter: 120,
    randomState: 42,
  }).fit(section.X);
  const mbkCenters = sortRowsLexicographic(mbk.clusterCenters_ as Matrix);
  const expectedMbkCenters = sortRowsLexicographic(section.minibatch_kmeans_centers);
  metrics.minibatch_kmeans_center_mse = meanSquaredErrorMatrixWithShapePenalty(
    mbkCenters,
    expectedMbkCenters,
  );
  const inertiaDiff = (mbk.inertia_ as number) - section.minibatch_kmeans_inertia;
  metrics.minibatch_kmeans_inertia_mse = inertiaDiff * inertiaDiff;

  const meanShift = new MeanShift({
    bandwidth: 1.2,
    maxIter: 50,
    clusterAll: true,
  }).fit(section.X);
  const meanShiftCenters = sortRowsLexicographic(meanShift.clusterCenters_ as Matrix);
  const expectedMeanShiftCenters = sortRowsLexicographic(section.meanshift_centers);
  metrics.meanshift_center_mse = meanSquaredErrorMatrixWithShapePenalty(
    meanShiftCenters,
    expectedMeanShiftCenters,
  );

  const affinity = new AffinityPropagation({
    damping: 0.7,
    maxIter: 300,
    convergenceIter: 30,
    randomState: 42,
  }).fit(section.X);
  const affinityCenters = sortRowsLexicographic(affinity.clusterCenters_ as Matrix);
  const expectedAffinityCenters = sortRowsLexicographic(section.affinity_centers);
  metrics.affinity_center_mse = meanSquaredErrorMatrixWithShapePenalty(
    affinityCenters,
    expectedAffinityCenters,
  );
}

{
  const section = fixture.additional_estimators.decomposition;
  const ipca = new IncrementalPCA({ nComponents: 2, batchSize: 2 });
  ipca.partialFit(section.incremental_pca_X.slice(0, 2));
  ipca.partialFit(section.incremental_pca_X.slice(2));
  const ipcaOut = ipca.transform(section.incremental_pca_X);
  metrics.incremental_pca_transform_distance_mse = meanSquaredErrorMatrix(
    pairwiseDistanceMatrix(ipcaOut),
    pairwiseDistanceMatrix(section.incremental_pca_transform),
  );

  const fa = new FactorAnalysis({ nComponents: 2 }).fit(section.factor_analysis_X);
  const faOut = fa.transform(section.factor_analysis_X);
  metrics.factor_analysis_latent_distance_mse = meanSquaredErrorMatrix(
    pairwiseDistanceMatrix(faOut),
    pairwiseDistanceMatrix(section.factor_analysis_latent),
  );

  const mbnmf = new MiniBatchNMF({
    nComponents: 2,
    batchSize: 2,
    maxIter: 120,
    randomState: 42,
  });
  const W = mbnmf.fitTransform(section.minibatch_nmf_X);
  const recon = mbnmf.inverseTransform(W);
  metrics.minibatch_nmf_reconstruction_mse = meanSquaredErrorMatrix(
    recon,
    section.minibatch_nmf_reconstruction,
  );
}

{
  const section = fixture.additional_estimators.multioutput;
  const moc = new MultiOutputClassifier(
    () => new DecisionTreeClassifier({ maxDepth: 3, randomState: 42 }),
  ).fit(section.X, section.Y_classifier);
  metrics.multioutput_classifier_mismatch = mismatchRateMatrix(
    moc.predict(section.X),
    section.multioutput_classifier_pred,
  );

  const cc = new ClassifierChain(
    () => new DecisionTreeClassifier({ maxDepth: 3, randomState: 42 }),
    { order: [1, 0] },
  ).fit(section.X, section.Y_classifier);
  metrics.classifier_chain_mismatch = mismatchRateMatrix(
    cc.predict(section.X),
    section.classifier_chain_pred,
  );

  const mor = new MultiOutputRegressor(
    () => new LinearRegression({ solver: "normal" }),
  ).fit(section.X, section.Y_regressor);
  metrics.multioutput_regressor_mse = meanSquaredErrorMatrix(
    mor.predict(section.X),
    section.multioutput_regressor_pred,
  );

  const rc = new RegressorChain(
    () => new LinearRegression({ solver: "normal" }),
    { order: [1, 0] },
  ).fit(section.X, section.Y_regressor);
  metrics.regressor_chain_mse = meanSquaredErrorMatrix(
    rc.predict(section.X),
    section.regressor_chain_pred,
  );
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
  composition_pipeline_transform_mse: threshold(
    "PARITY_MAX_COMPOSITION_PIPELINE_TRANSFORM_MSE",
    fixtureThresholds.composition_pipeline_transform_mse ?? 1e-10,
  ),
  composition_column_transformer_mse: threshold(
    "PARITY_MAX_COMPOSITION_COLUMN_TRANSFORMER_MSE",
    fixtureThresholds.composition_column_transformer_mse ?? 1e-10,
  ),
  splitter_group_shuffle_rate_mse: threshold(
    "PARITY_MAX_SPLITTER_GROUP_SHUFFLE_RATE_MSE",
    fixtureThresholds.splitter_group_shuffle_rate_mse ?? 0.12,
  ),
  splitter_stratified_group_rate_mse: threshold(
    "PARITY_MAX_SPLITTER_STRATIFIED_GROUP_RATE_MSE",
    fixtureThresholds.splitter_stratified_group_rate_mse ?? 0.12,
  ),
  inspection_permutation_mean_mse: threshold(
    "PARITY_MAX_INSPECTION_PERMUTATION_MEAN_MSE",
    fixtureThresholds.inspection_permutation_mean_mse ?? 0.08,
  ),
  inspection_permutation_rank_mismatch: threshold(
    "PARITY_MAX_INSPECTION_PERMUTATION_RANK_MISMATCH",
    fixtureThresholds.inspection_permutation_rank_mismatch ?? 0.5,
  ),
  minibatch_kmeans_center_mse: threshold(
    "PARITY_MAX_MINIBATCH_KMEANS_CENTER_MSE",
    fixtureThresholds.minibatch_kmeans_center_mse ?? 0.5,
  ),
  minibatch_kmeans_inertia_mse: threshold(
    "PARITY_MAX_MINIBATCH_KMEANS_INERTIA_MSE",
    fixtureThresholds.minibatch_kmeans_inertia_mse ?? 5,
  ),
  meanshift_center_mse: threshold(
    "PARITY_MAX_MEANSHIFT_CENTER_MSE",
    fixtureThresholds.meanshift_center_mse ?? 0.6,
  ),
  affinity_center_mse: threshold(
    "PARITY_MAX_AFFINITY_CENTER_MSE",
    fixtureThresholds.affinity_center_mse ?? 1.2,
  ),
  incremental_pca_transform_distance_mse: threshold(
    "PARITY_MAX_INCREMENTAL_PCA_TRANSFORM_DISTANCE_MSE",
    fixtureThresholds.incremental_pca_transform_distance_mse ?? 1e-8,
  ),
  factor_analysis_latent_distance_mse: threshold(
    "PARITY_MAX_FACTOR_ANALYSIS_LATENT_DISTANCE_MSE",
    fixtureThresholds.factor_analysis_latent_distance_mse ?? 0.7,
  ),
  minibatch_nmf_reconstruction_mse: threshold(
    "PARITY_MAX_MINIBATCH_NMF_RECONSTRUCTION_MSE",
    fixtureThresholds.minibatch_nmf_reconstruction_mse ?? 0.2,
  ),
  multioutput_classifier_mismatch: threshold(
    "PARITY_MAX_MULTIOUTPUT_CLASSIFIER_MISMATCH",
    fixtureThresholds.multioutput_classifier_mismatch ?? 0.35,
  ),
  classifier_chain_mismatch: threshold(
    "PARITY_MAX_CLASSIFIER_CHAIN_MISMATCH",
    fixtureThresholds.classifier_chain_mismatch ?? 0.4,
  ),
  multioutput_regressor_mse: threshold(
    "PARITY_MAX_MULTIOUTPUT_REGRESSOR_MSE",
    fixtureThresholds.multioutput_regressor_mse ?? 0.2,
  ),
  regressor_chain_mse: threshold(
    "PARITY_MAX_REGRESSOR_CHAIN_MSE",
    fixtureThresholds.regressor_chain_mse ?? 0.25,
  ),
};

let failed = false;
const failures: string[] = [];
for (const [name, value] of Object.entries(metrics)) {
  const limit = limits[name];
  const pass = value <= limit;
  const status = pass ? "PASS" : "FAIL";
  console.log(`${status} ${name}: value=${value} limit=${limit}`);
  if (!pass) {
    failed = true;
    failures.push(name);
  }
}

const reportPath = process.env.PARITY_SKLEARN_REPORT_PATH;
if (reportPath) {
  const absoluteReportPath = resolve(reportPath);
  mkdirSync(dirname(absoluteReportPath), { recursive: true });
  writeFileSync(
    absoluteReportPath,
    JSON.stringify(
      {
        generatedAt: new Date().toISOString(),
        failed,
        failures,
        metrics,
        limits,
      },
      null,
      2,
    ),
    "utf-8",
  );
}

if (failed) {
  console.error("sklearn parity check failed.");
  process.exit(1);
}

console.log("sklearn parity check passed.");
