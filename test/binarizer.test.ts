import { expect, test } from "bun:test";
import { Binarizer } from "../src/preprocessing/Binarizer";

test("Binarizer thresholds values with default threshold=0", () => {
  const X = [
    [-1, 0, 2],
    [3, -2, 0.5],
  ];

  const bin = new Binarizer();
  expect(bin.fitTransform(X)).toEqual([
    [0, 0, 1],
    [1, 0, 1],
  ]);
});

test("Binarizer supports custom threshold", () => {
  const X = [
    [0.2, 0.5, 0.8],
    [0.7, 0.4, 0.6],
  ];

  const bin = new Binarizer({ threshold: 0.6 });
  expect(bin.fitTransform(X)).toEqual([
    [0, 0, 1],
    [1, 0, 0],
  ]);
});
