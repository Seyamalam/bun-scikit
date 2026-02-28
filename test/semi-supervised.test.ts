import { expect, test } from "bun:test";
import {
  LabelPropagation,
  LabelSpreading,
} from "../src";

const X = [
  [0.0, 0.0],
  [0.1, 0.1],
  [0.2, 0.0],
  [2.0, 2.0],
  [2.1, 2.2],
  [1.9, 2.1],
];

const yPartiallyLabeled = [0, -1, -1, 1, -1, -1];

test("LabelPropagation learns transductive labels from partial supervision", () => {
  const model = new LabelPropagation({
    kernel: "rbf",
    gamma: 5,
    maxIter: 200,
    tolerance: 1e-4,
  }).fit(X, yPartiallyLabeled);

  expect(model.transduction_).not.toBeNull();
  expect(model.transduction_!.length).toBe(X.length);
  expect(model.predict([[0.15, 0.05], [2.05, 2.1]])).toEqual([0, 1]);
});

test("LabelSpreading learns transductive labels from partial supervision", () => {
  const model = new LabelSpreading({
    kernel: "knn",
    nNeighbors: 2,
    alpha: 0.3,
    maxIter: 200,
    tolerance: 1e-4,
  }).fit(X, yPartiallyLabeled);

  expect(model.transduction_).not.toBeNull();
  expect(model.transduction_!.length).toBe(X.length);
  expect(model.predict([[0.15, 0.05], [2.05, 2.1]])).toEqual([0, 1]);
});
