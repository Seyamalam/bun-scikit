# Getting Started

## Install

```bash
bun install bun-scikit
```

## Quick Start

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

## Local Development

```bash
bun install
bun run test
bun run typecheck
```
