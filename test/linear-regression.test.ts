import { expect, test } from "bun:test";
import { LinearRegression } from "../src/linear_model/LinearRegression";

test("LinearRegression (normal equation) fits a simple line", () => {
  const X = [[1], [2], [3], [4], [5]];
  const y = [3, 5, 7, 9, 11]; // y = 2x + 1

  const model = new LinearRegression({ solver: "normal" });
  model.fit(X, y);

  expect(model.intercept_).toBeCloseTo(1, 4);
  expect(model.coef_[0]).toBeCloseTo(2, 4);

  const prediction = model.predict([[6]])[0];
  expect(prediction).toBeCloseTo(13, 4);
});

test("LinearRegression (gradient descent) converges on simple data", () => {
  const X = [[0], [1], [2], [3], [4]];
  const y = [1, 3, 5, 7, 9];

  const model = new LinearRegression({
    solver: "gd",
    learningRate: 0.05,
    maxIter: 20_000,
    tolerance: 1e-10,
  });

  model.fit(X, y);

  expect(model.intercept_).toBeCloseTo(1, 2);
  expect(model.coef_[0]).toBeCloseTo(2, 2);
  expect(model.score(X, y)).toBeGreaterThan(0.999);
});
