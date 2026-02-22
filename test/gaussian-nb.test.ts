import { expect, test } from "bun:test";
import { GaussianNB } from "../src/naive_bayes/GaussianNB";

test("GaussianNB learns a separable Gaussian dataset", () => {
  const X = [
    [-2.2, -1.9],
    [-1.8, -2.1],
    [-2.0, -1.7],
    [1.8, 2.2],
    [2.1, 1.9],
    [1.7, 2.0],
  ];
  const y = [0, 0, 0, 1, 1, 1];

  const model = new GaussianNB();
  model.fit(X, y);

  expect(model.score(X, y)).toBeGreaterThan(0.99);
  expect(model.predict([[-2.1, -1.8], [2.0, 2.1]])).toEqual([0, 1]);
  const proba = model.predictProba([[0, 0]])[0];
  expect(proba[0] + proba[1]).toBeCloseTo(1, 10);
});
