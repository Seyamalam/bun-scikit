import { expect, test } from "bun:test";
import { OneHotEncoder } from "../src/preprocessing/OneHotEncoder";

test("OneHotEncoder encodes sorted categories deterministically", () => {
  const X = [
    [2, 10],
    [1, 10],
    [2, 20],
  ];

  const encoder = new OneHotEncoder();
  const encoded = encoder.fitTransform(X);

  expect(encoder.categories_).toEqual([
    [1, 2],
    [10, 20],
  ]);
  expect(encoder.nOutputFeatures_).toBe(4);
  expect(encoded).toEqual([
    [0, 1, 1, 0],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
  ]);
});

test("OneHotEncoder handleUnknown='ignore' leaves unknown categories as zero vectors", () => {
  const train = [
    [1],
    [2],
  ];
  const testX = [
    [2],
    [3],
  ];

  const encoder = new OneHotEncoder({ handleUnknown: "ignore" });
  encoder.fit(train);
  const encoded = encoder.transform(testX);

  expect(encoded).toEqual([
    [0, 1],
    [0, 0],
  ]);
});

test("OneHotEncoder throws on unknown category when handleUnknown='error'", () => {
  const encoder = new OneHotEncoder({ handleUnknown: "error" });
  encoder.fit([[1], [2]]);
  expect(() => encoder.transform([[3]])).toThrow(/unknown category/i);
});
