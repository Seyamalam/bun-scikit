import { expect, test } from "bun:test";
import { SGDClassifier } from "../src/linear_model/SGDClassifier";

test("SGDClassifier hinge loss learns separable labels", () => {
  const X = [[-3], [-2], [-1], [1], [2], [3]];
  const y = [0, 0, 0, 1, 1, 1];

  const model = new SGDClassifier({
    loss: "hinge",
    learningRate: 0.1,
    maxIter: 8_000,
    tolerance: 1e-7,
    l2: 0.001,
  });
  model.fit(X, y);

  expect(model.score(X, y)).toBeGreaterThan(0.99);
  expect(model.predict([[-0.2], [0.2]])).toEqual([0, 1]);
});

test("SGDClassifier log_loss exposes probabilities", () => {
  const X = [[0], [1], [2], [3], [4], [5]];
  const y = [0, 0, 0, 1, 1, 1];

  const model = new SGDClassifier({
    loss: "log_loss",
    learningRate: 0.8,
    maxIter: 3_000,
    tolerance: 1e-6,
  });
  model.fit(X, y);

  const proba = model.predictProba([[1.5], [4.5]]);
  expect(proba[0][1]).toBeLessThan(0.5);
  expect(proba[1][1]).toBeGreaterThan(0.5);
});
