import { expect, test } from "bun:test";
import { GradientBoostingRegressor } from "../src";

test("GradientBoostingRegressor fits a smooth nonlinear pattern", () => {
  const X = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]];
  const y = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81];

  const model = new GradientBoostingRegressor({
    nEstimators: 120,
    learningRate: 0.08,
    maxDepth: 2,
    randomState: 7,
  }).fit(X, y);

  expect(model.score(X, y)).toBeGreaterThan(0.97);
});
