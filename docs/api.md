# API Reference

## Preprocessing

### `StandardScaler`

- `fit(X: number[][]): this`
- `transform(X: number[][]): number[][]`
- `fitTransform(X: number[][]): number[][]`
- `inverseTransform(X: number[][]): number[][]`

Learned attributes:

- `mean_: number[] | null`
- `scale_: number[] | null`

## Linear Models

### `LinearRegression`

Constructor options:

- `fitIntercept?: boolean` (default `true`)
- `solver?: "normal" | "gd"` (default `"normal"`)
- `learningRate?: number` (default `0.01`)
- `maxIter?: number` (default `10000`)
- `tolerance?: number` (default `1e-8`)

Methods:

- `fit(X: number[][], y: number[]): this`
- `predict(X: number[][]): number[]`
- `score(X: number[][], y: number[]): number` (`R^2`)

Learned attributes:

- `coef_: number[]`
- `intercept_: number`

## Model Selection

### `trainTestSplit(X, y, options?)`

Options:

- `testSize?: number` (`0 < float < 1` or integer count)
- `shuffle?: boolean` (default `true`)
- `randomState?: number` (default `42`)

Returns:

- `XTrain`, `XTest`, `yTrain`, `yTest`

## Metrics

- `meanSquaredError(yTrue, yPred): number`
- `meanAbsoluteError(yTrue, yPred): number`
- `r2Score(yTrue, yPred): number`
