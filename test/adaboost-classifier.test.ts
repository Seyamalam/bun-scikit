import { expect, test } from "bun:test";
import { AdaBoostClassifier } from "../src";

test("AdaBoostClassifier learns a separable binary dataset", () => {
  const X = [[-3], [-2], [-1], [1], [2], [3], [4], [5]];
  const y = [0, 0, 0, 1, 1, 1, 1, 1];

  const model = new AdaBoostClassifier(null, {
    nEstimators: 30,
    learningRate: 0.8,
    randomState: 42,
  }).fit(X, y);

  const preds = model.predict(X);
  expect(preds.length).toBe(X.length);
  expect(model.score(X, y)).toBeGreaterThan(0.9);

  const proba = model.predictProba([[0], [4]]);
  expect(proba[0][1]).toBeLessThan(0.7);
  expect(proba[1][1]).toBeGreaterThan(0.5);
});
