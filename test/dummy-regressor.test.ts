import { expect, test } from "bun:test";
import { DummyRegressor } from "../src/dummy/DummyRegressor";

test("DummyRegressor mean strategy predicts training mean", () => {
  const X = [[1], [2], [3], [4]];
  const y = [1, 3, 5, 7];

  const model = new DummyRegressor({ strategy: "mean" }).fit(X, y);
  expect(model.constant_).toBeCloseTo(4, 10);
  expect(model.predict([[10], [20]])).toEqual([4, 4]);
});

test("DummyRegressor supports quantile and constant strategies", () => {
  const X = [[1], [2], [3], [4], [5]];
  const y = [2, 4, 6, 8, 10];

  const q = new DummyRegressor({ strategy: "quantile", quantile: 0.75 }).fit(X, y);
  expect(q.constant_).toBeCloseTo(8, 10);

  const c = new DummyRegressor({ strategy: "constant", constant: -5 }).fit(X, y);
  expect(c.predict([[99], [100]])).toEqual([-5, -5]);
});
