import { expect, test } from "bun:test";
import { StandardScaler } from "../src/preprocessing/StandardScaler";

test("StandardScaler normalizes each feature to zero mean", () => {
  const X = [
    [1, 10],
    [2, 20],
    [3, 30],
  ];

  const scaler = new StandardScaler();
  const transformed = scaler.fitTransform(X);

  const featureMeans = [0, 0];
  for (const row of transformed) {
    featureMeans[0] += row[0];
    featureMeans[1] += row[1];
  }

  featureMeans[0] /= transformed.length;
  featureMeans[1] /= transformed.length;

  expect(Math.abs(featureMeans[0])).toBeLessThan(1e-10);
  expect(Math.abs(featureMeans[1])).toBeLessThan(1e-10);
});

test("StandardScaler inverseTransform recovers original values", () => {
  const X = [
    [5, 100],
    [7, 120],
    [9, 140],
  ];
  const scaler = new StandardScaler();
  const transformed = scaler.fitTransform(X);
  const reconstructed = scaler.inverseTransform(transformed);

  for (let i = 0; i < X.length; i += 1) {
    for (let j = 0; j < X[i].length; j += 1) {
      expect(reconstructed[i][j]).toBeCloseTo(X[i][j], 8);
    }
  }
});
