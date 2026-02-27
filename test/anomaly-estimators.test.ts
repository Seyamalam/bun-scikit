import { expect, test } from "bun:test";
import {
  IsolationForest,
  LocalOutlierFactor,
  OneClassSVM,
} from "../src";

test("IsolationForest detects a distant outlier", () => {
  const X = [
    [0, 0],
    [0.1, 0.1],
    [-0.1, 0.1],
    [0.05, -0.05],
    [8, 8],
  ];
  const model = new IsolationForest({ contamination: 0.2, randomState: 7 }).fit(X);
  const pred = model.predict(X);
  expect(pred.slice(0, 4)).toEqual([1, 1, 1, 1]);
  expect(pred[4]).toBe(-1);
});

test("LocalOutlierFactor fits and scores novelty samples", () => {
  const X = [
    [0, 0],
    [0.1, 0.1],
    [-0.1, 0.1],
    [0.05, -0.05],
    [8, 8],
  ];
  const lof = new LocalOutlierFactor({
    nNeighbors: 2,
    contamination: 0.2,
    novelty: true,
  }).fit(X);
  const pred = lof.predict([[0.02, 0.01], [9, 9]]);
  expect(pred).toEqual([1, -1]);
});

test("OneClassSVM detects distant anomalies", () => {
  const X = [
    [0, 0],
    [0.1, 0.1],
    [-0.1, 0.1],
    [0.05, -0.05],
    [0.02, 0.0],
  ];
  const model = new OneClassSVM({ nu: 0.2, kernel: "rbf", gamma: "scale" }).fit(X);
  const pred = model.predict([[0.01, 0.01], [6, 6]]);
  expect(pred).toEqual([1, -1]);
});
