import { expect, test } from "bun:test";
import {
  LogisticRegression,
  RFECV,
  RFE,
  SelectFromModel,
  mutualInfoClassif,
  mutualInfoRegression,
} from "../src";

test("mutual information functions rank informative features higher", () => {
  const X = [
    [0, 7, 1],
    [0, 7, 2],
    [0, 7, 3],
    [1, 7, 4],
    [1, 7, 5],
    [1, 7, 6],
  ];
  const yClass = [0, 0, 0, 1, 1, 1];
  const yReg = [0, 0.1, 0.2, 1.1, 1.2, 1.3];

  const miClass = mutualInfoClassif(X, yClass, { nBins: 4 });
  const miReg = mutualInfoRegression(X, yReg, { nBins: 4 });
  expect(miClass[0]).toBeGreaterThan(miClass[1]);
  expect(miReg[0]).toBeGreaterThan(miReg[1]);
});

test("SelectFromModel, RFE, and RFECV select informative columns", () => {
  const X = [
    [-3, 0, 10],
    [-2, 0, 10],
    [-1, 0, 10],
    [1, 0, 10],
    [2, 0, 10],
    [3, 0, 10],
  ];
  const y = [0, 0, 0, 1, 1, 1];

  const select = new SelectFromModel(
    () =>
      new LogisticRegression({
        maxIter: 400,
        learningRate: 0.1,
        tolerance: 1e-6,
      }),
    { threshold: "mean" },
  ).fit(X, y);
  expect((select.getSupport(true) as number[]).includes(0)).toBe(true);

  const rfe = new RFE(
    () =>
      new LogisticRegression({
        maxIter: 400,
        learningRate: 0.1,
        tolerance: 1e-6,
      }),
    { nFeaturesToSelect: 1, step: 1 },
  ).fit(X, y);
  expect(rfe.getSupport(true)).toEqual([0]);

  const rfecv = new RFECV(
    () =>
      new LogisticRegression({
        maxIter: 400,
        learningRate: 0.1,
        tolerance: 1e-6,
      }),
    { cv: 3, minFeaturesToSelect: 1, step: 1 },
  ).fit(X, y);
  const support = rfecv.getSupport(true) as number[];
  expect(support.includes(0)).toBe(true);
  expect(rfecv.cvResults_.meanTestScore.length).toBeGreaterThan(0);
});
