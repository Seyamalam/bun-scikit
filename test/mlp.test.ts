import { expect, test } from "bun:test";
import {
  MLPClassifier,
  MLPRegressor,
} from "../src";

test("MLPClassifier fits linearly-separable data and predicts probabilities", () => {
  const X = [[-2], [-1], [1], [2]];
  const y = [0, 0, 1, 1];
  const clf = new MLPClassifier({
    hiddenLayerSizes: [4],
    activation: "tanh",
    maxIter: 600,
    learningRateInit: 0.05,
    randomState: 42,
    solver: "adam",
    tolerance: 1e-6,
  }).fit(X, y);

  const pred = clf.predict(X);
  expect(pred).toEqual(y);
  const proba = clf.predictProba([[0], [1.5]]);
  expect(proba.length).toBe(2);
  expect(proba[0].length).toBe(2);
});

test("MLPRegressor fits smooth nonlinear signal", () => {
  const X = [[-2], [-1], [0], [1], [2], [3]];
  const y = [4, 1, 0, 1, 4, 9];
  const reg = new MLPRegressor({
    hiddenLayerSizes: [8],
    activation: "tanh",
    maxIter: 1500,
    learningRateInit: 0.03,
    randomState: 7,
    solver: "adam",
    tolerance: 1e-6,
  }).fit(X, y);

  const score = reg.score(X, y);
  expect(score).toBeGreaterThan(0.95);
  const pred = reg.predict([[1.5], [2.5]]);
  expect(pred.length).toBe(2);
});
