import { expect, test } from "bun:test";
import { LinearSVC } from "../src/svm/LinearSVC";

test("LinearSVC learns a linearly separable boundary", () => {
  const X = [[-3], [-2], [-1], [1], [2], [3]];
  const y = [0, 0, 0, 1, 1, 1];

  const model = new LinearSVC({
    C: 1,
    learningRate: 0.1,
    maxIter: 8_000,
    tolerance: 1e-7,
  });
  model.fit(X, y);

  expect(model.score(X, y)).toBeGreaterThan(0.99);
  expect(model.predict([[-0.5], [0.5]])).toEqual([0, 1]);
});
