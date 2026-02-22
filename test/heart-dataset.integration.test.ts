import { expect, test } from "bun:test";
import {
  LinearRegression,
  StandardScaler,
  meanSquaredError,
  trainTestSplit,
} from "../src";
import { loadHeartDataset } from "../test_data/loadHeartDataset";

function mean(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

test("heart.csv workflow: split, scale, fit, and beat a baseline predictor", async () => {
  const { X, y } = await loadHeartDataset();
  expect(X.length).toBeGreaterThan(1000);
  expect(X.length).toBe(y.length);
  expect(X[0].length).toBe(13);

  const { XTrain, XTest, yTrain, yTest } = trainTestSplit(X, y, {
    testSize: 0.2,
    randomState: 42,
    shuffle: true,
  });

  const scaler = new StandardScaler();
  const XTrainScaled = scaler.fitTransform(XTrain);
  const XTestScaled = scaler.transform(XTest);

  const model = new LinearRegression({
    solver: "gd",
    learningRate: 0.03,
    maxIter: 30_000,
    tolerance: 1e-9,
  });

  model.fit(XTrainScaled, yTrain);
  const predictions = model.predict(XTestScaled);
  const modelMse = meanSquaredError(yTest, predictions);

  const baselineValue = mean(yTrain);
  const baselinePredictions = new Array(yTest.length).fill(baselineValue);
  const baselineMse = meanSquaredError(yTest, baselinePredictions);

  expect(Number.isFinite(modelMse)).toBe(true);
  expect(modelMse).toBeLessThan(baselineMse);
});
