import { expect, test } from "bun:test";
import { KNeighborsRegressor } from "../src";

test("KNeighborsRegressor fits smooth regression data", () => {
  const X = [[0], [1], [2], [3], [4], [5], [6]];
  const y = [0, 1, 4, 9, 16, 25, 36];

  const model = new KNeighborsRegressor({ nNeighbors: 2 }).fit(X, y);
  const pred = model.predict([[2.1], [4.2]]);
  expect(pred.length).toBe(2);
  expect(model.score(X, y)).toBeGreaterThan(0.85);
});

test("KNeighborsRegressor supports distance weighting", () => {
  const X = [[0], [1], [2], [10]];
  const y = [0, 1, 2, 100];
  const model = new KNeighborsRegressor({
    nNeighbors: 2,
    weights: "distance",
  }).fit(X, y);
  const [nearTwo] = model.predict([[2.05]]);
  expect(Math.abs(nearTwo - 2)).toBeLessThan(1);
});
