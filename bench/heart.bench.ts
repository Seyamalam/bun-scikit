import { LinearRegression, StandardScaler, meanSquaredError, r2Score, trainTestSplit } from "../src";
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

function runBenchmark(
  name: string,
  options: ConstructorParameters<typeof LinearRegression>[0],
): void {
  const model = new LinearRegression(options);
  const fitStart = performance.now();
  model.fit(XTrainScaled, yTrain);
  const fitMs = performance.now() - fitStart;

  const predictStart = performance.now();
  const predictions = model.predict(XTestScaled);
  const predictMs = performance.now() - predictStart;

  const mse = meanSquaredError(yTest, predictions);
  const r2 = r2Score(yTest, predictions);

  console.log(
    `${name} | fit=${fitMs.toFixed(2)}ms predict=${predictMs.toFixed(2)}ms mse=${mse.toFixed(6)} r2=${r2.toFixed(6)}`,
  );
}

console.log(
  `Heart dataset benchmark | samples=${X.length} features=${X[0].length} train=${XTrain.length} test=${XTest.length}`,
);

runBenchmark("linear_regression(normal)", { solver: "normal" });
runBenchmark("linear_regression(gd)", {
  solver: "gd",
  learningRate: 0.03,
  maxIter: 30_000,
  tolerance: 1e-9,
});
