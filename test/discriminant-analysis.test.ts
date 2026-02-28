import { expect, test } from "bun:test";
import {
  LinearDiscriminantAnalysis,
  QuadraticDiscriminantAnalysis,
} from "../src";

const X = [
  [0, 0],
  [0.1, 0.2],
  [-0.1, 0.1],
  [2, 2],
  [2.2, 1.9],
  [1.9, 2.1],
];
const y = [0, 0, 0, 1, 1, 1];

test("LinearDiscriminantAnalysis fits and predicts class probabilities", () => {
  const lda = new LinearDiscriminantAnalysis().fit(X, y);
  const proba = lda.predictProba([[0.05, 0.05], [2.05, 2.0]]);
  expect(proba.length).toBe(2);
  expect(proba[0].length).toBe(2);
  expect(lda.predict([[0.05, 0.05], [2.05, 2.0]])).toEqual([0, 1]);
});

test("QuadraticDiscriminantAnalysis fits and predicts", () => {
  const qda = new QuadraticDiscriminantAnalysis({ regParam: 0.01 }).fit(X, y);
  const pred = qda.predict([[0.05, 0.05], [2.05, 2.0]]);
  expect(pred).toEqual([0, 1]);
  expect(qda.score(X, y)).toBeGreaterThan(0.95);
});
