import { expect, test } from "bun:test";
import { MinMaxScaler } from "../src/preprocessing/MinMaxScaler";

test("MinMaxScaler scales each feature to [0, 1] by default", () => {
  const X = [
    [1, 10],
    [2, 20],
    [3, 30],
  ];

  const scaler = new MinMaxScaler();
  const transformed = scaler.fitTransform(X);

  expect(transformed[0][0]).toBeCloseTo(0, 10);
  expect(transformed[0][1]).toBeCloseTo(0, 10);
  expect(transformed[2][0]).toBeCloseTo(1, 10);
  expect(transformed[2][1]).toBeCloseTo(1, 10);
});

test("MinMaxScaler supports custom feature range and inverseTransform", () => {
  const X = [
    [5, 100],
    [7, 120],
    [9, 140],
  ];

  const scaler = new MinMaxScaler({ featureRange: [-1, 1] });
  const transformed = scaler.fitTransform(X);
  const reconstructed = scaler.inverseTransform(transformed);

  expect(transformed[0][0]).toBeCloseTo(-1, 10);
  expect(transformed[2][1]).toBeCloseTo(1, 10);

  for (let i = 0; i < X.length; i += 1) {
    for (let j = 0; j < X[i].length; j += 1) {
      expect(reconstructed[i][j]).toBeCloseTo(X[i][j], 8);
    }
  }
});

test("MinMaxScaler maps constant feature columns to lower bound", () => {
  const X = [
    [2, 10],
    [2, 20],
    [2, 30],
  ];

  const scaler = new MinMaxScaler({ featureRange: [0, 1] });
  const transformed = scaler.fitTransform(X);

  expect(transformed[0][0]).toBeCloseTo(0, 10);
  expect(transformed[1][0]).toBeCloseTo(0, 10);
  expect(transformed[2][0]).toBeCloseTo(0, 10);
});
