import { expect, test } from "bun:test";
import {
  IterativeImputer,
  MissingIndicator,
} from "../src";

test("IterativeImputer fills missing values", () => {
  const X = [
    [1, 2, Number.NaN],
    [2, Number.NaN, 6],
    [3, 6, 9],
    [4, 8, 12],
  ];
  const imputer = new IterativeImputer({ maxIter: 5, tolerance: 1e-6 }).fit(X);
  const transformed = imputer.transform(X);
  for (let i = 0; i < transformed.length; i += 1) {
    for (let j = 0; j < transformed[i].length; j += 1) {
      expect(Number.isFinite(transformed[i][j])).toBe(true);
    }
  }
});

test("MissingIndicator reports missing columns", () => {
  const X = [
    [1, Number.NaN, 3],
    [2, 5, Number.NaN],
    [3, 6, 7],
  ];
  const ind = new MissingIndicator({ features: "missing-only", errorOnNew: false }).fit(X);
  expect(ind.features_).toEqual([1, 2]);
  const mask = ind.transform(X);
  expect(mask.length).toBe(X.length);
  expect(mask[0]).toEqual([1, 0]);
  expect(mask[1]).toEqual([0, 1]);
});
