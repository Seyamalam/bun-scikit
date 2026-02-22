import { expect, test } from "bun:test";
import { RobustScaler } from "../src/preprocessing/RobustScaler";

test("RobustScaler centers by median and scales by IQR", () => {
  const X = [
    [1, 10],
    [2, 20],
    [3, 30],
    [100, 40],
  ];

  const scaler = new RobustScaler();
  const transformed = scaler.fitTransform(X);

  expect(transformed[1][0]).toBeCloseTo(-0.0196078431, 8);
  expect(transformed[2][0]).toBeCloseTo(0.0196078431, 8);
  expect(transformed[1][1]).toBeCloseTo(-0.3333333333, 8);
  expect(transformed[2][1]).toBeCloseTo(0.3333333333, 8);
});

test("RobustScaler inverseTransform reconstructs original data", () => {
  const X = [
    [5, 100],
    [7, 120],
    [9, 140],
  ];

  const scaler = new RobustScaler({ quantileRange: [10, 90] });
  const transformed = scaler.fitTransform(X);
  const reconstructed = scaler.inverseTransform(transformed);

  for (let i = 0; i < X.length; i += 1) {
    for (let j = 0; j < X[i].length; j += 1) {
      expect(reconstructed[i][j]).toBeCloseTo(X[i][j], 8);
    }
  }
});

test("RobustScaler supports disabling centering or scaling", () => {
  const X = [[1], [2], [3]];

  const noCenter = new RobustScaler({ withCentering: false, withScaling: true });
  const centered = noCenter.fitTransform(X);
  expect(centered[0][0]).toBeCloseTo(1, 10);

  const noScale = new RobustScaler({ withCentering: true, withScaling: false });
  const scaled = noScale.fitTransform(X);
  expect(scaled[1][0]).toBeCloseTo(0, 10);
});
