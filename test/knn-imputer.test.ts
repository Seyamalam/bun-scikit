import { expect, test } from "bun:test";
import { KNNImputer } from "../src/preprocessing/KNNImputer";

test("KNNImputer fills NaN values from nearest neighbors", () => {
  const X = [
    [1, 10],
    [2, Number.NaN],
    [3, 30],
    [4, 40],
  ];

  const imputer = new KNNImputer({ nNeighbors: 2, weights: "uniform" });
  const transformed = imputer.fitTransform(X);

  expect(transformed[1][1]).toBeCloseTo(20, 6);
  expect(transformed[0]).toEqual([1, 10]);
  expect(transformed[3]).toEqual([4, 40]);
});

test("KNNImputer falls back to feature mean when neighbors are unavailable", () => {
  const train = [
    [1, Number.NaN],
    [2, 8],
    [3, 10],
  ];
  const imputer = new KNNImputer({ nNeighbors: 1 }).fit(train);
  const transformed = imputer.transform([[Number.NaN, Number.NaN]]);
  expect(transformed[0][1]).toBeCloseTo(9, 6);
});
