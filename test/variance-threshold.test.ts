import { expect, test } from "bun:test";
import { VarianceThreshold } from "../src/feature_selection/VarianceThreshold";

test("VarianceThreshold drops zero-variance columns", () => {
  const X = [
    [1, 10, 5],
    [1, 20, 6],
    [1, 30, 7],
  ];
  const selector = new VarianceThreshold({ threshold: 0 });
  const transformed = selector.fitTransform(X);
  expect(selector.selectedFeatureIndices_).toEqual([1, 2]);
  expect(transformed).toEqual([
    [10, 5],
    [20, 6],
    [30, 7],
  ]);
});
