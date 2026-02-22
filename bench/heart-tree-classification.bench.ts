import {
  DecisionTreeClassifier,
  RandomForestClassifier,
  StandardScaler,
  accuracyScore,
  f1Score,
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

function benchmarkModel(label: string, fitAndPredict: () => number[]): void {
  const start = performance.now();
  const predictions = fitAndPredict();
  const elapsedMs = performance.now() - start;
  const accuracy = accuracyScore(yTest, predictions);
  const f1 = f1Score(yTest, predictions);
  console.log(
    `${label} | total=${elapsedMs.toFixed(2)}ms accuracy=${accuracy.toFixed(6)} f1=${f1.toFixed(6)}`,
  );
}

console.log(
  `Heart tree benchmark | samples=${X.length} features=${X[0].length} train=${XTrain.length} test=${XTest.length}`,
);

benchmarkModel("decision_tree(maxDepth=8)", () => {
  const model = new DecisionTreeClassifier({
    maxDepth: 8,
    minSamplesLeaf: 3,
    randomState: 42,
  });
  model.fit(XTrainScaled, yTrain);
  return model.predict(XTestScaled);
});

benchmarkModel("random_forest(nEstimators=80,maxDepth=8)", () => {
  const model = new RandomForestClassifier({
    nEstimators: 80,
    maxDepth: 8,
    minSamplesLeaf: 2,
    randomState: 42,
  });
  model.fit(XTrainScaled, yTrain);
  return model.predict(XTestScaled);
});
