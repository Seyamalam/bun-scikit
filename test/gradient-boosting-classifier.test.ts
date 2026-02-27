import { expect, test } from "bun:test";
import { GradientBoostingClassifier } from "../src";

test("GradientBoostingClassifier learns a binary boundary", () => {
  const X = [[-4], [-3], [-2], [-1], [1], [2], [3], [4], [5], [6]];
  const y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1];

  const model = new GradientBoostingClassifier({
    nEstimators: 120,
    learningRate: 0.08,
    maxDepth: 2,
    randomState: 13,
  }).fit(X, y);

  expect(model.score(X, y)).toBeGreaterThan(0.9);
  const proba = model.predictProba([[-3], [5]]);
  expect(proba[0][1]).toBeLessThan(0.7);
  expect(proba[1][1]).toBeGreaterThan(0.5);
});
