import { expect, test } from "bun:test";
import { RandomForestClassifier } from "../src/ensemble/RandomForestClassifier";

test("RandomForestClassifier learns separable clusters", () => {
  const X = [
    [0.0, 0.1],
    [0.2, 0.1],
    [0.1, 0.2],
    [1.0, 1.0],
    [1.1, 0.9],
    [0.9, 1.1],
  ];
  const y = [0, 0, 0, 1, 1, 1];

  const model = new RandomForestClassifier({
    nEstimators: 25,
    maxDepth: 4,
    randomState: 42,
  });
  model.fit(X, y);

  const predictions = model.predict(X);
  expect(predictions).toEqual(y);
  expect(model.score(X, y)).toBe(1);
});
