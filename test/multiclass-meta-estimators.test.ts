import { expect, test } from "bun:test";
import {
  KNeighborsClassifier,
  OneVsOneClassifier,
  OneVsRestClassifier,
} from "../src";

const X = [
  [0.0, 0.1],
  [0.2, -0.1],
  [-0.2, 0.0],
  [3.0, 3.1],
  [2.9, 3.2],
  [3.2, 2.9],
  [6.0, 6.2],
  [5.8, 6.0],
  [6.1, 5.9],
];
const y = [0, 0, 0, 1, 1, 1, 2, 2, 2];

test("OneVsRestClassifier supports multiclass fit/predict/proba", () => {
  const model = new OneVsRestClassifier(
    () => new KNeighborsClassifier({ nNeighbors: 1 }),
    { normalizeProba: true },
  ).fit(X, y);

  const pred = model.predict(X);
  expect(pred).toEqual(y);
  const proba = model.predictProba([[0.1, 0.0], [3.1, 3.0], [6.0, 6.1]]);
  expect(proba[0].length).toBe(3);
});

test("OneVsOneClassifier supports multiclass fit/predict/proba", () => {
  const model = new OneVsOneClassifier(
    () => new KNeighborsClassifier({ nNeighbors: 1 }),
  ).fit(X, y);

  const pred = model.predict(X);
  expect(pred).toEqual(y);
  const proba = model.predictProba([[0.1, 0.0], [3.1, 3.0], [6.0, 6.1]]);
  expect(proba[0].length).toBe(3);
});
