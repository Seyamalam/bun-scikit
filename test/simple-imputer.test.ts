import { expect, test } from "bun:test";
import { SimpleImputer } from "../src/preprocessing/SimpleImputer";

test("SimpleImputer mean strategy fills NaN values", () => {
  const X = [
    [1, Number.NaN],
    [2, 4],
    [3, 6],
  ];

  const imputer = new SimpleImputer({ strategy: "mean" });
  const transformed = imputer.fitTransform(X);

  expect(imputer.statistics_).toEqual([2, 5]);
  expect(transformed).toEqual([
    [1, 5],
    [2, 4],
    [3, 6],
  ]);
});

test("SimpleImputer supports median/most_frequent/constant", () => {
  const X = [
    [1, 10],
    [Number.NaN, 10],
    [3, Number.NaN],
    [4, 30],
  ];

  const median = new SimpleImputer({ strategy: "median" }).fitTransform(X);
  const mode = new SimpleImputer({ strategy: "most_frequent" }).fitTransform(X);
  const constant = new SimpleImputer({ strategy: "constant", fillValue: -1 }).fitTransform(X);

  expect(median).toEqual([
    [1, 10],
    [3, 10],
    [3, 10],
    [4, 30],
  ]);

  expect(mode).toEqual([
    [1, 10],
    [1, 10],
    [3, 10],
    [4, 30],
  ]);

  expect(constant).toEqual([
    [1, 10],
    [-1, 10],
    [3, -1],
    [4, 30],
  ]);
});

test("SimpleImputer throws when feature is entirely missing with non-constant strategy", () => {
  const X = [
    [Number.NaN, 1],
    [Number.NaN, 2],
  ];
  expect(() => new SimpleImputer({ strategy: "mean" }).fit(X)).toThrow(/only missing values/i);
});
