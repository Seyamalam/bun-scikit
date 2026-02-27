import { expect, test } from "bun:test";
import {
  LinearRegression,
  LogisticRegression,
  permutationImportance,
} from "../src";

test("permutationImportance is deterministic with fixed randomState", () => {
  const X = [
    [0, 10],
    [1, 11],
    [2, 9],
    [3, 12],
    [4, 8],
    [5, 13],
  ];
  const y = [1, 3, 5, 7, 9, 11];

  const model = new LinearRegression().fit(X, y);
  const a = permutationImportance(model, X, y, {
    scoring: "r2",
    nRepeats: 8,
    randomState: 7,
  });
  const b = permutationImportance(model, X, y, {
    scoring: "r2",
    nRepeats: 8,
    randomState: 7,
  });
  expect(a).toEqual(b);
});

test("permutationImportance highlights informative features", () => {
  const X = [
    [0, 0],
    [1, 0],
    [2, 1],
    [3, 0],
    [4, 1],
    [5, 0],
    [6, 1],
    [7, 0],
  ];
  const y = [0, 0, 0, 1, 1, 1, 1, 1];
  const model = new LogisticRegression({
    maxIter: 200,
    learningRate: 0.2,
    tolerance: 1e-6,
  }).fit(X, y);

  const result = permutationImportance(model, X, y, {
    scoring: "accuracy",
    nRepeats: 10,
    randomState: 11,
  });

  expect(result.importancesMean.length).toBe(2);
  expect(result.importancesMean[0]).toBeGreaterThan(result.importancesMean[1]);
  expect(result.importancesStd[0]).toBeGreaterThanOrEqual(0);
});
