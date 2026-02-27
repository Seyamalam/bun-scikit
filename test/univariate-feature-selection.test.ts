import { expect, test } from "bun:test";
import {
  SelectKBest,
  SelectPercentile,
  chi2,
  f_classif,
  f_regression,
} from "../src";

test("chi2 validates non-negative inputs and ranks informative features", () => {
  const X = [
    [10, 1, 0],
    [9, 1, 0],
    [8, 1, 0],
    [1, 1, 4],
    [0, 1, 5],
    [1, 1, 4],
  ];
  const y = [0, 0, 0, 1, 1, 1];
  const [scores, pValues] = chi2(X, y);
  expect(scores.length).toBe(3);
  expect(pValues.length).toBe(3);
  expect(scores[0]).toBeGreaterThan(scores[1]);
  expect(scores[2]).toBeGreaterThan(scores[1]);
  expect(() => chi2([[1, -1], [1, 2]], [0, 1])).toThrow(/non-negative/i);
});

test("f_classif and f_regression produce finite scores with sensible ordering", () => {
  const XClass = [
    [-3, 1, 5],
    [-2, 1, 4],
    [-1, 1, 6],
    [1, 1, 5],
    [2, 1, 4],
    [3, 1, 6],
  ];
  const yClass = [0, 0, 0, 1, 1, 1];
  const [fClassifScores, fClassifP] = f_classif(XClass, yClass);
  expect(fClassifScores[0]).toBeGreaterThan(fClassifScores[1]);
  expect(fClassifP[0]).toBeLessThan(fClassifP[1]);

  const XReg = [
    [0, 1, 7],
    [1, 1, 8],
    [2, 1, 9],
    [3, 1, 10],
    [4, 1, 11],
    [5, 1, 12],
  ];
  const yReg = [0, 2, 4, 6, 8, 10];
  const [fRegScores, fRegP] = f_regression(XReg, yReg);
  expect(fRegScores[0]).toBeGreaterThan(fRegScores[1]);
  expect(fRegP[0]).toBeLessThan(fRegP[1]);
});

test("SelectKBest and SelectPercentile select expected features", () => {
  const X = [
    [-3, 0, 5],
    [-2, 0, 5],
    [-1, 0, 5],
    [1, 0, 5],
    [2, 0, 5],
    [3, 0, 5],
  ];
  const y = [0, 0, 0, 1, 1, 1];

  const selectKBest = new SelectKBest({ k: 1, scoreFunc: f_classif }).fit(X, y);
  expect(selectKBest.getSupport(true)).toEqual([0]);
  expect(selectKBest.transform(X)[0].length).toBe(1);

  const selectPercentile = new SelectPercentile({
    percentile: 34,
    scoreFunc: f_classif,
  }).fit(X, y);
  const selected = selectPercentile.getSupport(true) as number[];
  expect(selected.length).toBe(2);
  expect(selected.includes(0)).toBe(true);
});
