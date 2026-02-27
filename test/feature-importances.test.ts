import { expect, test } from "bun:test";
import {
  AdaBoostClassifier,
  DecisionTreeClassifier,
  DecisionTreeRegressor,
  GradientBoostingClassifier,
  GradientBoostingRegressor,
  HistGradientBoostingClassifier,
  HistGradientBoostingRegressor,
  RandomForestClassifier,
  RandomForestRegressor,
} from "../src";

function sum(values: number[] | null | undefined): number {
  if (!values) {
    return 0;
  }
  let total = 0;
  for (let i = 0; i < values.length; i += 1) {
    total += values[i];
  }
  return total;
}

function withJsTreeBackend(run: () => void): void {
  const previous = process.env.BUN_SCIKIT_TREE_BACKEND;
  process.env.BUN_SCIKIT_TREE_BACKEND = "js";
  try {
    run();
  } finally {
    if (previous === undefined) {
      delete process.env.BUN_SCIKIT_TREE_BACKEND;
    } else {
      process.env.BUN_SCIKIT_TREE_BACKEND = previous;
    }
  }
}

test("tree and forest estimators expose normalized feature importances", () => {
  const X = [
    [0, 2],
    [1, 2],
    [2, 2],
    [3, 2],
    [4, 2],
    [5, 2],
    [6, 2],
    [7, 2],
  ];
  const yCls = [0, 0, 0, 0, 1, 1, 1, 1];
  const yReg = [0, 1, 2, 3, 10, 11, 12, 13];

  withJsTreeBackend(() => {
    const dtc = new DecisionTreeClassifier({ maxDepth: 3, randomState: 42 }).fit(X, yCls);
    expect(dtc.featureImportances_).toBeTruthy();
    expect(sum(dtc.featureImportances_)).toBeCloseTo(1, 6);

    const dtr = new DecisionTreeRegressor({ maxDepth: 3, randomState: 42 }).fit(X, yReg);
    expect(dtr.featureImportances_).toBeTruthy();
    expect(sum(dtr.featureImportances_)).toBeCloseTo(1, 6);

    const rfc = new RandomForestClassifier({
      nEstimators: 20,
      maxDepth: 4,
      randomState: 42,
    }).fit(X, yCls);
    expect(rfc.featureImportances_).toBeTruthy();
    expect(sum(rfc.featureImportances_)).toBeCloseTo(1, 6);

    const rfr = new RandomForestRegressor({
      nEstimators: 20,
      maxDepth: 4,
      randomState: 42,
    }).fit(X, yReg);
    expect(rfr.featureImportances_).toBeTruthy();
    expect(sum(rfr.featureImportances_)).toBeCloseTo(1, 6);
  });
});

test("boosting estimators expose normalized feature importances", () => {
  const X = [
    [0, 2],
    [1, 2],
    [2, 2],
    [3, 2],
    [4, 2],
    [5, 2],
    [6, 2],
    [7, 2],
    [8, 2],
    [9, 2],
  ];
  const yCls = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
  const yReg = [0, 1, 2, 3, 4, 8, 9, 10, 11, 12];

  withJsTreeBackend(() => {
    const ada = new AdaBoostClassifier(null, { nEstimators: 20, randomState: 42 }).fit(X, yCls);
    expect(ada.featureImportances_).toBeTruthy();
    expect(sum(ada.featureImportances_)).toBeCloseTo(1, 6);

    const gbc = new GradientBoostingClassifier({
      nEstimators: 30,
      learningRate: 0.1,
      maxDepth: 2,
      randomState: 42,
    }).fit(X, yCls);
    expect(gbc.featureImportances_).toBeTruthy();
    expect(sum(gbc.featureImportances_)).toBeCloseTo(1, 6);

    const gbr = new GradientBoostingRegressor({
      nEstimators: 30,
      learningRate: 0.1,
      maxDepth: 2,
      randomState: 42,
    }).fit(X, yReg);
    expect(gbr.featureImportances_).toBeTruthy();
    expect(sum(gbr.featureImportances_)).toBeCloseTo(1, 6);

    const hgbc = new HistGradientBoostingClassifier({
      maxIter: 40,
      learningRate: 0.1,
      minSamplesLeaf: 1,
      earlyStopping: false,
      randomState: 42,
    }).fit(X, yCls);
    expect(hgbc.featureImportances_).toBeTruthy();
    expect(sum(hgbc.featureImportances_)).toBeCloseTo(1, 6);

    const hgbr = new HistGradientBoostingRegressor({
      maxIter: 40,
      learningRate: 0.1,
      minSamplesLeaf: 1,
      earlyStopping: false,
      randomState: 42,
    }).fit(X, yReg);
    expect(hgbr.featureImportances_).toBeTruthy();
    expect(sum(hgbr.featureImportances_)).toBeCloseTo(1, 6);
  });
});
