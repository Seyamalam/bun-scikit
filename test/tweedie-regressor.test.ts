import { expect, test } from "bun:test";
import { TweedieRegressor } from "../src";

test("TweedieRegressor fits positive targets for compound power values", () => {
  const X = [[0], [1], [2], [3], [4], [5]];
  const y = [1.1, 1.4, 1.9, 2.6, 3.4, 4.6];

  const reg = new TweedieRegressor({
    power: 1.5,
    alpha: 0,
    learningRate: 0.02,
    maxIter: 4000,
    tolerance: 1e-7,
  }).fit(X, y);

  const pred = reg.predict([[2.5], [6]]);
  expect(pred[0]).toBeGreaterThan(1.5);
  expect(pred[0]).toBeLessThan(3.5);
  expect(pred[1]).toBeGreaterThan(pred[0]);
});

test("TweedieRegressor supports gaussian power=0 mode", () => {
  const X = [[0], [1], [2], [3], [4], [5]];
  const y = [1, 3, 5, 7, 9, 11];

  const reg = new TweedieRegressor({
    power: 0,
    alpha: 0,
    learningRate: 0.05,
    maxIter: 4000,
    tolerance: 1e-7,
  }).fit(X, y);

  expect(reg.predict([[6]])[0]).toBeGreaterThan(11);
  expect(reg.score(X, y)).toBeGreaterThan(0.99);
});
