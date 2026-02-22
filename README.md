# bun-scikit

`bun-scikit` is a scikit-learn-inspired machine learning library for Bun + TypeScript.

## Scope (v1)

- `StandardScaler`
- `LinearRegression` (`normal` and `gd` solvers)
- `trainTestSplit`
- Regression metrics: `meanSquaredError`, `meanAbsoluteError`, `r2Score`

## Install

```bash
bun install
```

## Usage

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
const preds = model.predict(XTest);

console.log("MSE:", meanSquaredError(yTest, preds));
```

## Scripts

```bash
bun run test
bun run typecheck
bun run bench
```
