import { expect, test } from "bun:test";
import {
  HistGradientBoostingClassifier,
  HistGradientBoostingRegressor,
} from "../src";

test("HistGradientBoostingRegressor fits nonlinear regression", () => {
  const X = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]];
  const y = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81];

  const model = new HistGradientBoostingRegressor({
    maxIter: 150,
    learningRate: 0.08,
    maxBins: 16,
    minSamplesLeaf: 1,
    randomState: 42,
  }).fit(X, y);

  expect(model.score(X, y)).toBeGreaterThan(0.95);
});

test("HistGradientBoostingClassifier fits binary classification", () => {
  const X = [[-4], [-3], [-2], [-1], [1], [2], [3], [4], [5], [6]];
  const y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1];

  const model = new HistGradientBoostingClassifier({
    maxIter: 120,
    learningRate: 0.08,
    maxBins: 16,
    minSamplesLeaf: 1,
    randomState: 42,
  }).fit(X, y);

  expect(model.score(X, y)).toBeGreaterThan(0.9);
  const proba = model.predictProba([[-3], [5]]);
  expect(proba[0][1]).toBeLessThan(0.7);
  expect(proba[1][1]).toBeGreaterThan(0.5);
});

test("HistGradientBoosting defaults stay constant on tiny datasets", () => {
  const XReg = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]];
  const yReg = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81];
  const reg = new HistGradientBoostingRegressor({ randomState: 42 }).fit(XReg, yReg);
  const pred = reg.predict([[1.5], [3.5], [7.5]]);
  expect(pred).toEqual([28.5, 28.5, 28.5]);

  const XCls = [[-4], [-3], [-2], [-1], [1], [2], [3], [4], [5], [6]];
  const yCls = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
  const cls = new HistGradientBoostingClassifier({ randomState: 42 }).fit(XCls, yCls);
  expect(cls.predictProba([[-3], [5]])).toEqual([
    [0.4, 0.6],
    [0.4, 0.6],
  ]);
});
