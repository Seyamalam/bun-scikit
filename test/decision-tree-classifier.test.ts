import { expect, test } from "bun:test";
import { DecisionTreeClassifier } from "../src/tree/DecisionTreeClassifier";

test("DecisionTreeClassifier fits a separable threshold dataset", () => {
  const X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [2, 2],
    [2, 3],
    [3, 2],
  ];
  const y = [0, 0, 0, 1, 1, 1];

  const model = new DecisionTreeClassifier({
    maxDepth: 3,
    randomState: 42,
  });
  model.fit(X, y);

  expect(model.predict(X)).toEqual(y);
  expect(model.score(X, y)).toBe(1);
});
