# Getting Started

## Install

```bash
bun install bun-scikit
```

## Quick Start

Build native kernels first:

```bash
bun run native:build
```

```ts
import {
  LinearRegression,
  StandardScaler,
  meanSquaredError,
  trainTestSplit,
} from "bun-scikit";

const X = [
  [1, 2],
  [2, 3],
  [3, 4],
  [4, 5],
];
const y = [5, 7, 9, 11];

const scaler = new StandardScaler();
const XScaled = scaler.fitTransform(X);
const { XTrain, XTest, yTrain, yTest } = trainTestSplit(XScaled, y, {
  testSize: 0.25,
  randomState: 42,
});

const model = new LinearRegression();
model.fit(XTrain, yTrain);
const predictions = model.predict(XTest);

console.log("MSE:", meanSquaredError(yTest, predictions));
```

## Binary Classification Example

```ts
import {
  LogisticRegression,
  StandardScaler,
  accuracyScore,
  trainTestSplit,
} from "bun-scikit";

const X = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
];
const y = [0, 0, 0, 1];

const scaler = new StandardScaler();
const XScaled = scaler.fitTransform(X);
const { XTrain, XTest, yTrain, yTest } = trainTestSplit(XScaled, y, {
  testSize: 0.25,
  randomState: 42,
});

const classifier = new LogisticRegression();
classifier.fit(XTrain, yTrain);
const predictions = classifier.predict(XTest);

console.log("Accuracy:", accuracyScore(yTest, predictions));
```

## Zig Backend Smoke Check

Use this quick check after install to verify tree/forest are running on Zig:

```ts
import { DecisionTreeClassifier, RandomForestClassifier } from "bun-scikit";

const X = [
  [0, 0],
  [0, 1],
  [1, 0],
  [2, 2],
  [2, 3],
  [3, 2],
];
const y = [0, 0, 0, 1, 1, 1];

const tree = new DecisionTreeClassifier({ maxDepth: 3, randomState: 42 });
tree.fit(X, y);
console.log("DecisionTree backend:", tree.fitBackend_, tree.fitBackendLibrary_);

const forest = new RandomForestClassifier({ nEstimators: 25, maxDepth: 4, randomState: 42 });
forest.fit(X, y);
console.log("RandomForest backend:", forest.fitBackend_, forest.fitBackendLibrary_);
```

Expected output should report `zig` for both `fitBackend_` values.

## Local Development

```bash
bun install
bun run test
bun run typecheck
```
