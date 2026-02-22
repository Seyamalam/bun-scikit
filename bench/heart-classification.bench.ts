import {
  accuracyScore,
  f1Score,
  KNeighborsClassifier,
  LogisticRegression,
  StandardScaler,
  trainTestSplit,
} from "../src";
import { loadHeartDataset } from "../test_data/loadHeartDataset";

const { X, y } = await loadHeartDataset();
const { XTrain, XTest, yTrain, yTest } = trainTestSplit(X, y, {
  testSize: 0.2,
  randomState: 42,
  shuffle: true,
});

const scaler = new StandardScaler();
const XTrainScaled = scaler.fitTransform(XTrain);
const XTestScaled = scaler.transform(XTest);

function benchmarkModel(
  label: string,
  fitAndPredict: () => number[],
): void {
  const fitStart = performance.now();
  const predictions = fitAndPredict();
  const elapsed = performance.now() - fitStart;
  const accuracy = accuracyScore(yTest, predictions);
  const f1 = f1Score(yTest, predictions);
  console.log(
    `${label} | total=${elapsed.toFixed(2)}ms accuracy=${accuracy.toFixed(6)} f1=${f1.toFixed(6)}`,
  );
}

console.log(
  `Heart classification benchmark | samples=${X.length} features=${X[0].length} train=${XTrain.length} test=${XTest.length}`,
);

benchmarkModel("logistic_regression(gd)", () => {
  const model = new LogisticRegression({
    learningRate: 0.35,
    maxIter: 500,
    tolerance: 1e-5,
    l2: 0.01,
  });
  model.fit(XTrainScaled, yTrain);
  console.log(`  -> fit backend=${model.fitBackend_}`);
  return model.predict(XTestScaled);
});

benchmarkModel("k_neighbors_classifier(k=7)", () => {
  const model = new KNeighborsClassifier({ nNeighbors: 7 });
  model.fit(XTrainScaled, yTrain);
  return model.predict(XTestScaled);
});
