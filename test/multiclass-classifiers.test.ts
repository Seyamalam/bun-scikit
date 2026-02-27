import { expect, test } from "bun:test";
import { GaussianNB, KNeighborsClassifier, LinearSVC, SGDClassifier } from "../src";

const X = [
  [0.0, 0.0],
  [0.1, 0.2],
  [0.2, -0.1],
  [2.0, 2.1],
  [2.2, 1.9],
  [1.8, 2.0],
  [4.0, 4.2],
  [3.9, 4.1],
  [4.2, 3.8],
];
const y = [0, 0, 0, 1, 1, 1, 2, 2, 2];

test("GaussianNB supports multiclass probability predictions", () => {
  const model = new GaussianNB().fit(X, y);
  const preds = model.predict(X);
  expect(preds).toEqual(y);
  const proba = model.predictProba([[0.1, 0.1], [2.1, 2.0], [4.1, 4.0]]);
  expect(proba[0].length).toBe(3);
  for (let i = 0; i < proba.length; i += 1) {
    expect(proba[i][0] + proba[i][1] + proba[i][2]).toBeCloseTo(1, 8);
  }
});

test("KNeighborsClassifier supports multiclass", () => {
  const model = new KNeighborsClassifier({ nNeighbors: 3 }).fit(X, y);
  expect(model.predict(X)).toEqual(y);
  const proba = model.predictProba([[0.15, 0.0]]);
  expect(proba[0].length).toBe(3);
});

test("SGDClassifier supports multiclass with hinge and log_loss", () => {
  const hinge = new SGDClassifier({
    loss: "hinge",
    learningRate: 0.05,
    maxIter: 15_000,
    tolerance: 1e-7,
  }).fit(X, y);
  expect(hinge.score(X, y)).toBeGreaterThan(0.95);

  const logLoss = new SGDClassifier({
    loss: "log_loss",
    learningRate: 0.1,
    maxIter: 15_000,
    tolerance: 1e-7,
  }).fit(X, y);
  const proba = logLoss.predictProba([[0.1, 0.1], [2.1, 2.0], [4.1, 4.0]]);
  expect(proba[0].length).toBe(3);
});

test("LinearSVC supports multiclass", () => {
  const model = new LinearSVC({
    C: 1.0,
    learningRate: 0.05,
    maxIter: 15_000,
    tolerance: 1e-7,
  }).fit(X, y);
  const preds = model.predict(X);
  expect(new Set(preds).size).toBeGreaterThan(1);
  expect(model.score(X, y)).toBeGreaterThan(0.6);
});
