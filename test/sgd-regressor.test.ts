import { expect, test } from "bun:test";
import { SGDRegressor } from "../src/linear_model/SGDRegressor";

test("SGDRegressor fits a simple linear trend", () => {
  const X = [[0], [1], [2], [3], [4], [5]];
  const y = [1, 3, 5, 7, 9, 11];

  const model = new SGDRegressor({
    learningRate: 0.1,
    maxIter: 20_000,
    tolerance: 1e-8,
    l2: 0,
  });
  model.fit(X, y);

  expect(model.intercept_).toBeCloseTo(1, 2);
  expect(model.coef_[0]).toBeCloseTo(2, 2);
  expect(model.score(X, y)).toBeGreaterThan(0.999);
});
