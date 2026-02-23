import { expect, test } from "bun:test";
import { MaxAbsScaler } from "../src/preprocessing/MaxAbsScaler";

test("MaxAbsScaler scales each feature by maximum absolute value", () => {
  const X = [
    [-2, 10],
    [1, -5],
    [2, 0],
  ];

  const scaler = new MaxAbsScaler();
  const transformed = scaler.fitTransform(X);

  expect(scaler.maxAbs_).toEqual([2, 10]);
  expect(transformed).toEqual([
    [-1, 1],
    [0.5, -0.5],
    [1, 0],
  ]);
});

test("MaxAbsScaler inverseTransform reconstructs original values", () => {
  const X = [
    [3, -4],
    [6, 8],
  ];

  const scaler = new MaxAbsScaler();
  const transformed = scaler.fitTransform(X);
  const reconstructed = scaler.inverseTransform(transformed);

  for (let i = 0; i < X.length; i += 1) {
    for (let j = 0; j < X[i].length; j += 1) {
      expect(reconstructed[i][j]).toBeCloseTo(X[i][j], 10);
    }
  }
});
