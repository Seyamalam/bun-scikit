import { expect, test } from "bun:test";
import { Normalizer } from "../src/preprocessing/Normalizer";

test("Normalizer l2 normalizes each row to unit norm", () => {
  const X = [
    [3, 4],
    [1, 2],
  ];
  const transformed = new Normalizer({ norm: "l2" }).fitTransform(X);
  for (const row of transformed) {
    const norm = Math.hypot(...row);
    expect(norm).toBeCloseTo(1, 8);
  }
});

test("Normalizer l1 normalizes each row to absolute sum 1", () => {
  const X = [[1, -1, 2]];
  const transformed = new Normalizer({ norm: "l1" }).fitTransform(X);
  const absSum = transformed[0].reduce((sum, value) => sum + Math.abs(value), 0);
  expect(absSum).toBeCloseTo(1, 8);
});
