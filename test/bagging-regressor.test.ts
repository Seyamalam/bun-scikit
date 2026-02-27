import { expect, test } from "bun:test";
import { BaggingRegressor, DecisionTreeRegressor } from "../src";

test("BaggingRegressor improves fit on nonlinear signal", () => {
  const X = [[0], [1], [2], [3], [4], [5], [6], [7]];
  const y = [0, 1, 4, 9, 16, 25, 36, 49];

  const model = new BaggingRegressor(
    () => new DecisionTreeRegressor({ maxDepth: 4, randomState: 7 }),
    { nEstimators: 20, randomState: 11 },
  ).fit(X, y);

  expect(model.estimators_.length).toBe(20);
  expect(model.score(X, y)).toBeGreaterThan(0.9);
});
