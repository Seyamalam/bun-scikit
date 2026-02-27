import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import {
  CalibratedClassifierCV,
  DecisionTreeClassifier,
  GaussianNB,
  HistGradientBoostingClassifier,
  HistGradientBoostingRegressor,
  KFold,
  KernelPCA,
  KNeighborsClassifier,
  LogisticRegression,
  NMF,
  Pipeline,
  RandomForestClassifier,
  StandardScaler,
  VotingClassifier,
  crossValPredict,
} from "../src";

function meanAbsDiff(a: number[][], b: number[][]): number {
  let total = 0;
  let count = 0;
  for (let i = 0; i < a.length; i += 1) {
    for (let j = 0; j < a[i].length; j += 1) {
      total += Math.abs(a[i][j] - b[i][j]);
      count += 1;
    }
  }
  return total / count;
}

function meanSquaredError(a: number[][], b: number[][]): number {
  let total = 0;
  let count = 0;
  for (let i = 0; i < a.length; i += 1) {
    for (let j = 0; j < a[i].length; j += 1) {
      const diff = a[i][j] - b[i][j];
      total += diff * diff;
      count += 1;
    }
  }
  return total / count;
}

const fixture = JSON.parse(
  readFileSync(resolve("test/fixtures/sklearn-snapshots.json"), "utf-8"),
);

test("GaussianNB probabilities stay close to sklearn snapshot", () => {
  const model = new GaussianNB().fit(fixture.multiclass.X, fixture.multiclass.y);
  const proba = model.predictProba(fixture.multiclass.probe);
  expect(meanAbsDiff(proba, fixture.multiclass.gaussian_nb_proba)).toBeLessThan(0.12);
});

test("VotingClassifier soft probabilities stay close to sklearn snapshot", () => {
  const model = new VotingClassifier(
    [
      ["gnb", () => new GaussianNB()],
      ["knn", () => new KNeighborsClassifier({ nNeighbors: 3 })],
    ],
    { voting: "soft" },
  ).fit(fixture.multiclass.X, fixture.multiclass.y);

  const proba = model.predictProba(fixture.multiclass.probe);
  expect(meanAbsDiff(proba, fixture.multiclass.voting_soft_proba)).toBeLessThan(0.15);
});

test("CalibratedClassifierCV probabilities stay close to sklearn snapshot", () => {
  const model = new CalibratedClassifierCV(() => new GaussianNB(), {
    cv: 3,
    method: "sigmoid",
    ensemble: false,
    randomState: 42,
  }).fit(fixture.multiclass.X, fixture.multiclass.y);

  const proba = model.predictProba(fixture.multiclass.probe);
  expect(meanAbsDiff(proba, fixture.multiclass.calibrated_sigmoid_proba)).toBeLessThan(0.2);
});

test("NMF reconstruction stays close to sklearn snapshot", () => {
  const model = new NMF({
    nComponents: 2,
    maxIter: 500,
    tolerance: 1e-6,
    randomState: 42,
  });
  const W = model.fitTransform(fixture.nmf.X);
  const reconstruction = model.inverseTransform(W);
  expect(meanSquaredError(reconstruction, fixture.nmf.reconstruction)).toBeLessThan(0.05);
});

test("DecisionTreeClassifier predictions stay close to sklearn snapshot", () => {
  const previousTreeBackend = process.env.BUN_SCIKIT_TREE_BACKEND;
  process.env.BUN_SCIKIT_TREE_BACKEND = "js";
  try {
    const model = new DecisionTreeClassifier({ maxDepth: 4, randomState: 42 }).fit(
      fixture.multiclass.X,
      fixture.multiclass.y,
    );
    expect(model.predict(fixture.multiclass.X)).toEqual(fixture.multiclass.decision_tree_pred);
  } finally {
    if (previousTreeBackend === undefined) {
      delete process.env.BUN_SCIKIT_TREE_BACKEND;
    } else {
      process.env.BUN_SCIKIT_TREE_BACKEND = previousTreeBackend;
    }
  }
});

test("RandomForestClassifier predictions stay close to sklearn snapshot", () => {
  const previousTreeBackend = process.env.BUN_SCIKIT_TREE_BACKEND;
  process.env.BUN_SCIKIT_TREE_BACKEND = "js";
  try {
    const model = new RandomForestClassifier({
      nEstimators: 40,
      maxDepth: 4,
      randomState: 42,
    }).fit(fixture.multiclass.X, fixture.multiclass.y);

    const preds = model.predict(fixture.multiclass.X);
    let mismatches = 0;
    for (let i = 0; i < preds.length; i += 1) {
      if (preds[i] !== fixture.multiclass.random_forest_pred[i]) {
        mismatches += 1;
      }
    }
    expect(mismatches / preds.length).toBeLessThan(0.15);
  } finally {
    if (previousTreeBackend === undefined) {
      delete process.env.BUN_SCIKIT_TREE_BACKEND;
    } else {
      process.env.BUN_SCIKIT_TREE_BACKEND = previousTreeBackend;
    }
  }
});

test("HistGradientBoostingClassifier probabilities stay close to sklearn snapshot", () => {
  const model = new HistGradientBoostingClassifier({
    maxIter: 120,
    learningRate: 0.08,
    maxBins: 16,
    randomState: 42,
  }).fit(fixture.hist_gradient_boosting.X_binary, fixture.hist_gradient_boosting.y_binary);

  const proba = model.predictProba(fixture.hist_gradient_boosting.probe_binary);
  expect(meanAbsDiff(proba, fixture.hist_gradient_boosting.classifier_probe_proba)).toBeLessThan(
    0.25,
  );
});

test("HistGradientBoostingRegressor predictions stay close to sklearn snapshot", () => {
  const model = new HistGradientBoostingRegressor({
    maxIter: 150,
    learningRate: 0.08,
    maxBins: 16,
    randomState: 42,
  }).fit(fixture.hist_gradient_boosting.X_reg, fixture.hist_gradient_boosting.y_reg);

  const pred = model.predict(fixture.hist_gradient_boosting.probe_reg);
  let mse = 0;
  for (let i = 0; i < pred.length; i += 1) {
    const diff = pred[i] - fixture.hist_gradient_boosting.regressor_probe_pred[i];
    mse += diff * diff;
  }
  mse /= pred.length;
  expect(mse).toBeLessThan(50);
});

test("KernelPCA pairwise distances stay close to sklearn snapshot", () => {
  const model = new KernelPCA({
    nComponents: 2,
    kernel: "rbf",
    gamma: 0.5,
  }).fit(fixture.kernel_pca.X);

  const transformed = model.transform(fixture.kernel_pca.X);
  const a = transformed;
  const b = fixture.kernel_pca.train_transform;
  const distance = (X: number[][]): number[][] => {
    const out: number[][] = Array.from({ length: X.length }, () =>
      new Array<number>(X.length).fill(0),
    );
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
  };
  expect(meanSquaredError(distance(a), distance(b))).toBeLessThan(0.2);
});

test("Pipeline LogisticRegression probabilities stay close to sklearn snapshot", () => {
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

  const proba = model.predictProba(section.probe);
  expect(meanAbsDiff(proba, section.probe_proba)).toBeLessThan(0.15);
});

test("crossValPredict matches sklearn fixture within mismatch tolerance", () => {
  const section = fixture.pipeline_logistic_regression;
  const pred = crossValPredict(
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
  ) as number[];

  let mismatch = 0;
  for (let i = 0; i < pred.length; i += 1) {
    if (pred[i] !== section.cv_predict_kfold4[i]) {
      mismatch += 1;
    }
  }
  expect(mismatch / pred.length).toBeLessThan(0.2);
});
