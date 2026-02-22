import { expect, test } from "bun:test";
import {
  accuracyScore,
  DecisionTreeClassifier,
  f1Score,
  KNeighborsClassifier,
  LinearRegression,
  LogisticRegression,
  RandomForestClassifier,
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

test("heart.csv classification workflow beats majority baseline", async () => {
  const { X, y } = await loadHeartDataset();
  const { XTrain, XTest, yTrain, yTest } = trainTestSplit(X, y, {
    testSize: 0.2,
    randomState: 42,
    shuffle: true,
  });

  const scaler = new StandardScaler();
  const XTrainScaled = scaler.fitTransform(XTrain);
  const XTestScaled = scaler.transform(XTest);

  const positiveCount = yTrain.reduce((count, label) => count + (label === 1 ? 1 : 0), 0);
  const majorityLabel = positiveCount * 2 >= yTrain.length ? 1 : 0;
  const baselinePredictions = new Array(yTest.length).fill(majorityLabel);
  const baselineAccuracy = accuracyScore(yTest, baselinePredictions);

  const logistic = new LogisticRegression({
    learningRate: 0.2,
    maxIter: 3_000,
    tolerance: 1e-6,
    l2: 0.01,
  });
  logistic.fit(XTrainScaled, yTrain);
  const logisticPredictions = logistic.predict(XTestScaled);
  const logisticAccuracy = accuracyScore(yTest, logisticPredictions);
  const logisticF1 = f1Score(yTest, logisticPredictions);

  const knn = new KNeighborsClassifier({ nNeighbors: 7 });
  knn.fit(XTrainScaled, yTrain);
  const knnPredictions = knn.predict(XTestScaled);
  const knnAccuracy = accuracyScore(yTest, knnPredictions);

  const decisionTree = new DecisionTreeClassifier({
    maxDepth: 8,
    minSamplesLeaf: 3,
    randomState: 42,
  });
  decisionTree.fit(XTrainScaled, yTrain);
  const decisionTreeAccuracy = accuracyScore(yTest, decisionTree.predict(XTestScaled));

  const randomForest = new RandomForestClassifier({
    nEstimators: 80,
    maxDepth: 8,
    minSamplesLeaf: 2,
    randomState: 42,
  });
  randomForest.fit(XTrainScaled, yTrain);
  const randomForestAccuracy = accuracyScore(yTest, randomForest.predict(XTestScaled));

  expect(logisticAccuracy).toBeGreaterThan(baselineAccuracy);
  expect(logisticF1).toBeGreaterThan(0.75);
  expect(knnAccuracy).toBeGreaterThan(baselineAccuracy);
  expect(decisionTreeAccuracy).toBeGreaterThan(baselineAccuracy);
  expect(randomForestAccuracy).toBeGreaterThan(baselineAccuracy);
});
