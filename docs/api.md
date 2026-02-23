# API Reference

## Core Types

- `Vector`
- `Matrix`
- `RegressionModel`
- `ClassificationModel`

## Preprocessing

### `StandardScaler`

Methods:

- `fit(X: number[][]): this`
- `transform(X: number[][]): number[][]`
- `fitTransform(X: number[][]): number[][]`
- `inverseTransform(X: number[][]): number[][]`

Learned attributes:

- `mean_: number[] | null`
- `scale_: number[] | null`

## Linear Models

### `LinearRegressionOptions`

- `fitIntercept?: boolean`
- `solver?: "normal" | "gd"`
- `learningRate?: number`
- `maxIter?: number`
- `tolerance?: number`

### `LinearRegression`

Methods:

- `fit(X: number[][], y: number[]): this`
- `predict(X: number[][]): number[]`
- `score(X: number[][], y: number[]): number`

Learned attributes:

- `coef_: number[]`
- `intercept_: number`

### `LogisticRegressionOptions`

- `fitIntercept?: boolean`
- `learningRate?: number`
- `maxIter?: number`
- `tolerance?: number`
- `l2?: number`
- `backend?: "auto" | "js" | "zig"`

### `LogisticRegression`

Methods:

- `fit(X: number[][], y: number[]): this`
- `predict(X: number[][]): number[]`
- `predictProba(X: number[][]): number[][]`
- `score(X: number[][], y: number[]): number`

Learned attributes:

- `coef_: number[]`
- `intercept_: number`
- `classes_: number[]`
- `fitBackend_: "js" | "zig"`
- `fitBackendLibrary_: string | null`

## Tree Models

### `MaxFeaturesOption`

- `"sqrt" | "log2" | number | null`

### `DecisionTreeClassifierOptions`

- `maxDepth?: number`
- `minSamplesSplit?: number`
- `minSamplesLeaf?: number`
- `maxFeatures?: MaxFeaturesOption`
- `randomState?: number`

### `DecisionTreeClassifier`

Methods:

- `fit(X: number[][], y: number[]): this`
- `predict(X: number[][]): number[]`
- `score(X: number[][], y: number[]): number`

Learned attributes:

- `classes_: number[]`

## Ensemble Models

### `RandomForestClassifierOptions`

- `nEstimators?: number`
- `maxDepth?: number`
- `minSamplesSplit?: number`
- `minSamplesLeaf?: number`
- `maxFeatures?: MaxFeaturesOption`
- `bootstrap?: boolean`
- `randomState?: number`

### `RandomForestClassifier`

Methods:

- `fit(X: number[][], y: number[]): this`
- `predict(X: number[][]): number[]`
- `score(X: number[][], y: number[]): number`

Learned attributes:

- `classes_: number[]`

## Neighbors

### `KNeighborsClassifierOptions`

- `nNeighbors?: number`

### `KNeighborsClassifier`

Methods:

- `fit(X: number[][], y: number[]): this`
- `predict(X: number[][]): number[]`
- `score(X: number[][], y: number[]): number`

Learned attributes:

- `classes_: number[]`

## Model Selection

### `TrainTestSplitOptions`

- `testSize?: number`
- `shuffle?: boolean`
- `randomState?: number`

### `TrainTestSplitResult<TX, TY>`

- `XTrain: TX[]`
- `XTest: TX[]`
- `yTrain: TY[]`
- `yTest: TY[]`

### `trainTestSplit`

- `trainTestSplit<TX, TY>(X: TX[], y: TY[], options?: TrainTestSplitOptions): TrainTestSplitResult<TX, TY>`

## Regression Metrics

- `meanSquaredError(yTrue, yPred): number`
- `meanAbsoluteError(yTrue, yPred): number`
- `r2Score(yTrue, yPred): number`

## Classification Metrics

- `accuracyScore(yTrue, yPred): number`
- `precisionScore(yTrue, yPred, positiveLabel?): number`
- `recallScore(yTrue, yPred, positiveLabel?): number`
- `f1Score(yTrue, yPred, positiveLabel?): number`
- `confusionMatrix(yTrue, yPred, positiveLabel?): [[number, number], [number, number]]`
