import { expect, test } from "bun:test";
import { KNeighborsClassifier } from "../src/neighbors/KNeighborsClassifier";

test("KNeighborsClassifier predicts local majority class", () => {
  const X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [3, 3],
    [3, 4],
    [4, 3],
  ];
  const y = [0, 0, 0, 1, 1, 1];

  const model = new KNeighborsClassifier({ nNeighbors: 3 });
  model.fit(X, y);

  const preds = model.predict([
    [0.2, 0.1],
    [3.2, 3.1],
  ]);

  expect(preds).toEqual([0, 1]);
  expect(model.score(X, y)).toBeGreaterThan(0.99);
});
