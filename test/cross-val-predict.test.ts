import { expect, test } from "bun:test";
import {
  KFold,
  LogisticRegression,
  crossValPredict,
} from "../src";

test("crossValPredict returns out-of-fold label predictions", () => {
  const X = [[-2], [-1], [-0.5], [0.5], [1], [2], [2.5], [3]];
  const y = [0, 0, 0, 1, 1, 1, 1, 1];

  const pred = crossValPredict(
    () => new LogisticRegression({ maxIter: 200, learningRate: 0.2, tolerance: 1e-5 }),
    X,
    y,
    { cv: new KFold({ nSplits: 4, shuffle: false }) },
  ) as number[];

  expect(pred.length).toBe(y.length);
  let matches = 0;
  for (let i = 0; i < y.length; i += 1) {
    if (pred[i] === y[i]) {
      matches += 1;
    }
  }
  expect(matches / y.length).toBeGreaterThan(0.75);
});

test("crossValPredict supports predictProba mode", () => {
  const X = [[-2], [-1], [-0.5], [0.5], [1], [2], [2.5], [3]];
  const y = [0, 0, 0, 1, 1, 1, 1, 1];

  const proba = crossValPredict(
    () => new LogisticRegression({ maxIter: 200, learningRate: 0.2, tolerance: 1e-5 }),
    X,
    y,
    { cv: 3, method: "predictProba" },
  ) as number[][];

  expect(proba.length).toBe(y.length);
  expect(proba[0].length).toBe(2);
  for (let i = 0; i < proba.length; i += 1) {
    const sum = proba[i][0] + proba[i][1];
    expect(sum).toBeCloseTo(1, 8);
  }
});
