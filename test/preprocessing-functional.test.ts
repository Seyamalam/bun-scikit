import { expect, test } from "bun:test";
import {
  addDummyFeature,
  binarize,
  maxabsScale,
  minmaxScale,
  robustScale,
  scale,
} from "../src";

test("addDummyFeature prepends a constant column", () => {
  expect(addDummyFeature([[2, 3], [4, 5]])).toEqual([
    [1, 2, 3],
    [1, 4, 5],
  ]);
  expect(addDummyFeature([[2, 3]], { value: -2 })).toEqual([[-2, 2, 3]]);
});

test("binarize applies thresholding without constructing an estimator", () => {
  expect(binarize([[-1, 0.2], [0.7, 0]], { threshold: 0 })).toEqual([
    [0, 1],
    [1, 0],
  ]);
});

test("functional scaler helpers mirror fitted transformer behavior", () => {
  const X = [[1, -2], [2, 0], [3, 4]];

  expect(scale(X)).toHaveLength(3);
  expect(minmaxScale(X, { featureRange: [-1, 1] })[0]).toEqual([-1, -1]);
  expect(maxabsScale(X)).toEqual([
    [1 / 3, -0.5],
    [2 / 3, 0],
    [1, 1],
  ]);

  const robust = robustScale(X);
  expect(robust).toHaveLength(3);
  expect(robust[1][0]).toBeCloseTo(0, 8);
});
