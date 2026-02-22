import { expect, test } from "bun:test";
import { FeatureUnion } from "../src/pipeline/FeatureUnion";
import { MinMaxScaler } from "../src/preprocessing/MinMaxScaler";
import { PolynomialFeatures } from "../src/preprocessing/PolynomialFeatures";

test("FeatureUnion concatenates transformer outputs", () => {
  const X = [
    [1, 2],
    [3, 4],
  ];

  const union = new FeatureUnion([
    ["poly", new PolynomialFeatures({ degree: 2, includeBias: false })],
    ["scale", new MinMaxScaler()],
  ]);

  const transformed = union.fitTransform(X);
  // poly(deg2,bias=false) for 2 features -> 5 columns, MinMax -> 2 columns.
  expect(transformed.length).toBe(2);
  expect(transformed[0].length).toBe(7);
  expect(transformed[0].slice(0, 5)).toEqual([1, 2, 1, 2, 4]);
  expect(transformed[0].slice(5)).toEqual([0, 0]);
  expect(transformed[1].slice(5)).toEqual([1, 1]);
});

test("FeatureUnion rejects duplicate names", () => {
  expect(
    () =>
      new FeatureUnion([
        ["dup", new MinMaxScaler()],
        ["dup", new PolynomialFeatures()],
      ]).fit([[1], [2]]),
  ).toThrow(/must be unique/i);
});
