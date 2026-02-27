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
- `solver?: "normal"`

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
- `fitBackend_: "zig"`
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
- `fitBackend_: "zig" | "js"`
- `fitBackendLibrary_: string | null`

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
- `fitBackend_: "zig" | "js"`
- `fitBackendLibrary_: string | null`

### `VotingClassifierOptions`

- `voting?: "hard" | "soft"`
- `weights?: number[]`

### `VotingClassifier`

Methods:

- `fit(X: number[][], y: number[]): this`
- `predict(X: number[][]): number[]`
- `predictProba(X: number[][]): number[][]`
- `score(X: number[][], y: number[]): number`

### `StackingClassifierOptions`

- `cv?: number`
- `passthrough?: boolean`
- `stackMethod?: "auto" | "predictProba" | "predict"`
- `randomState?: number`

### `StackingClassifier`

Methods:

- `fit(X: number[][], y: number[]): this`
- `predict(X: number[][]): number[]`
- `predictProba(X: number[][]): number[][]`
- `score(X: number[][], y: number[]): number`

### `AdaBoostClassifierOptions`

- `nEstimators?: number`
- `learningRate?: number`
- `randomState?: number`

### `AdaBoostClassifier`

Methods:

- `fit(X: number[][], y: number[]): this`
- `decisionFunction(X: number[][]): number[]`
- `predict(X: number[][]): number[]`
- `predictProba(X: number[][]): number[][]`
- `score(X: number[][], y: number[]): number`

Learned attributes:

- `classes_: number[]`
- `estimators_: unknown[]`
- `estimatorWeights_: number[]`

### `GradientBoostingClassifierOptions`

- `nEstimators?: number`
- `learningRate?: number`
- `maxDepth?: number`
- `minSamplesSplit?: number`
- `minSamplesLeaf?: number`
- `subsample?: number`
- `randomState?: number`

### `GradientBoostingClassifier`

Methods:

- `fit(X: number[][], y: number[]): this`
- `decisionFunction(X: number[][]): number[]`
- `predict(X: number[][]): number[]`
- `predictProba(X: number[][]): number[][]`
- `score(X: number[][], y: number[]): number`

Learned attributes:

- `classes_: number[]`
- `estimators_: unknown[]`
- `init_: number | null`

### `GradientBoostingRegressorOptions`

- `nEstimators?: number`
- `learningRate?: number`
- `maxDepth?: number`
- `minSamplesSplit?: number`
- `minSamplesLeaf?: number`
- `subsample?: number`
- `randomState?: number`

### `GradientBoostingRegressor`

Methods:

- `fit(X: number[][], y: number[]): this`
- `predict(X: number[][]): number[]`
- `score(X: number[][], y: number[]): number`

Learned attributes:

- `estimators_: unknown[]`
- `init_: number | null`

### `VotingRegressorOptions`

- `weights?: number[]`

### `VotingRegressor`

Methods:

- `fit(X: number[][], y: number[]): this`
- `predict(X: number[][]): number[]`
- `score(X: number[][], y: number[]): number`

Types:

- `VotingRegressorEstimatorSpec`

### `StackingRegressorOptions`

- `cv?: number`
- `passthrough?: boolean`
- `randomState?: number`

### `StackingRegressor`

Methods:

- `fit(X: number[][], y: number[]): this`
- `predict(X: number[][]): number[]`
- `score(X: number[][], y: number[]): number`

Types:

- `StackingRegressorEstimatorSpec`

### `BaggingClassifierOptions`

- `nEstimators?: number`
- `maxSamples?: number`
- `maxFeatures?: number`
- `bootstrap?: boolean`
- `bootstrapFeatures?: boolean`
- `randomState?: number`

### `BaggingClassifier`

Methods:

- `fit(X: number[][], y: number[]): this`
- `predict(X: number[][]): number[]`
- `predictProba(X: number[][]): number[][]`
- `score(X: number[][], y: number[]): number`

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

## Clustering

### `KMeansOptions`

- `nClusters?: number`
- `nInit?: number`
- `maxIter?: number`
- `tolerance?: number`
- `randomState?: number`

### `KMeans`

Methods:

- `fit(X: number[][]): this`
- `predict(X: number[][]): number[]`
- `fitPredict(X: number[][]): number[]`
- `transform(X: number[][]): number[][]`
- `score(X: number[][]): number`

Learned attributes:

- `clusterCenters_: number[][] | null`
- `labels_: number[] | null`
- `inertia_: number | null`
- `nIter_: number | null`
- `nFeaturesIn_: number | null`

### `DBSCANOptions`

- `eps?: number`
- `minSamples?: number`

### `DBSCAN`

Methods:

- `fit(X: number[][]): this`
- `fitPredict(X: number[][]): number[]`

Learned attributes:

- `labels_: number[] | null`
- `coreSampleIndices_: number[] | null`
- `components_: number[][] | null`
- `nFeaturesIn_: number | null`
- `nClusters_: number` (getter)

### `AgglomerativeClusteringOptions`

- `nClusters?: number`
- `linkage?: "ward" | "complete" | "average" | "single"`
- `metric?: "euclidean"`

### `AgglomerativeClustering`

Methods:

- `fit(X: number[][]): this`
- `fitPredict(X: number[][]): number[]`

Learned attributes:

- `labels_: number[] | null`
- `children_: number[][] | null`
- `distances_: number[] | null`
- `nConnectedComponents_: number`
- `nLeaves_: number | null`
- `nClusters_: number | null`
- `nFeaturesIn_: number | null`
- `nMerges_: number` (getter)

## Decomposition

### `PCAOptions`

- `nComponents?: number`
- `whiten?: boolean`
- `tolerance?: number`
- `maxIter?: number`

### `PCA`

Methods:

- `fit(X: number[][]): this`
- `transform(X: number[][]): number[][]`
- `fitTransform(X: number[][]): number[][]`
- `inverseTransform(X: number[][]): number[][]`

Learned attributes:

- `components_: number[][] | null`
- `explainedVariance_: number[] | null`
- `explainedVarianceRatio_: number[] | null`
- `singularValues_: number[] | null`
- `mean_: number[] | null`
- `nFeaturesIn_: number | null`

### `TruncatedSVDOptions`

- `nComponents?: number`
- `nIter?: number`
- `tolerance?: number`
- `randomState?: number`

### `TruncatedSVD`

Methods:

- `fit(X: number[][]): this`
- `transform(X: number[][]): number[][]`
- `fitTransform(X: number[][]): number[][]`
- `inverseTransform(X: number[][]): number[][]`

Learned attributes:

- `components_: number[][] | null`
- `explainedVariance_: number[] | null`
- `explainedVarianceRatio_: number[] | null`
- `singularValues_: number[] | null`
- `nFeaturesIn_: number | null`
- `nComponents_: number | null`

### `FastICAOptions`

- `nComponents?: number`
- `maxIter?: number`
- `tolerance?: number`
- `randomState?: number`

### `FastICA`

Methods:

- `fit(X: number[][]): this`
- `transform(X: number[][]): number[][]`
- `fitTransform(X: number[][]): number[][]`
- `inverseTransform(X: number[][]): number[][]`

Learned attributes:

- `components_: number[][] | null`
- `mixing_: number[][] | null`
- `mean_: number[] | null`
- `nIter_: number | null`
- `nFeaturesIn_: number | null`
- `nComponents_: number | null`

### `NMFOptions`

- `nComponents?: number`
- `maxIter?: number`
- `tolerance?: number`
- `randomState?: number`

### `NMF`

Methods:

- `fit(X: number[][]): this`
- `transform(X: number[][]): number[][]`
- `fitTransform(X: number[][]): number[][]`
- `inverseTransform(W: number[][]): number[][]`

Learned attributes:

- `components_: number[][] | null`
- `reconstructionErr_: number | null`
- `nIter_: number | null`
- `nFeaturesIn_: number | null`
- `nComponents_: number | null`

## Calibration

### `CalibratedClassifierCVOptions`

- `cv?: number`
- `method?: "sigmoid" | "isotonic"`
- `ensemble?: boolean`
- `randomState?: number`

### `CalibratedClassifierCV`

Methods:

- `fit(X: number[][], y: number[]): this`
- `predict(X: number[][]): number[]`
- `predictProba(X: number[][]): number[][]`
- `score(X: number[][], y: number[]): number`

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

## API Symbol Index

`AdaBoostClassifier`, `AdaBoostClassifierOptions`, `AgglomerativeClustering`, `AgglomerativeClusteringOptions`, `AgglomerativeLinkage`, `AgglomerativeMetric`, `BaggingClassifier`, `BaggingClassifierOptions`, `balancedAccuracyScore`, `Binarizer`, `BinarizerOptions`, `brierScoreLoss`, `BuiltInScoring`, `CalibratedClassifierCV`, `CalibratedClassifierCVOptions`, `CalibrationMethod`, `classificationReport`, `ClassificationReportLabelMetrics`, `ClassificationReportResult`, `ColumnSelector`, `ColumnTransformer`, `ColumnTransformerOptions`, `ColumnTransformerSpec`, `confusionMatrix`, `ConfusionMatrixResult`, `CrossValEstimator`, `crossValScore`, `CrossValScoreOptions`, `CrossValSplitter`, `DBSCAN`, `DBSCANOptions`, `DecisionTreeRegressor`, `DecisionTreeRegressorOptions`, `DummyClassifier`, `DummyClassifierOptions`, `DummyClassifierStrategy`, `DummyRegressor`, `DummyRegressorOptions`, `DummyRegressorStrategy`, `explainedVarianceScore`, `FastICA`, `FastICAOptions`, `FeatureUnion`, `FeatureUnionSpec`, `FoldIndices`, `GaussianNB`, `GaussianNBOptions`, `GradientBoostingClassifier`, `GradientBoostingClassifierOptions`, `GradientBoostingRegressor`, `GradientBoostingRegressorOptions`, `GridSearchCV`, `GridSearchCVOptions`, `GridSearchResultRow`, `ImputerStrategy`, `KFold`, `KFoldOptions`, `KMeans`, `KMeansOptions`, `LabelEncoder`, `LinearSVC`, `LinearSVCOptions`, `logLoss`, `matthewsCorrcoef`, `MaxAbsScaler`, `meanAbsolutePercentageError`, `MinMaxScaler`, `MinMaxScalerOptions`, `NMF`, `NMFOptions`, `Normalizer`, `NormalizerOptions`, `OneHotEncoder`, `OneHotEncoderOptions`, `PCA`, `PCAOptions`, `ParamDistributions`, `ParamGrid`, `Pipeline`, `PipelineStep`, `PolynomialFeatures`, `PolynomialFeaturesOptions`, `RandomForestRegressor`, `RandomForestRegressorOptions`, `RandomizedSearchCV`, `RandomizedSearchCVOptions`, `RandomizedSearchResultRow`, `RepeatedKFold`, `RepeatedKFoldOptions`, `RepeatedStratifiedKFold`, `RepeatedStratifiedKFoldOptions`, `RobustScaler`, `RobustScalerOptions`, `rocAucScore`, `ScoringFn`, `SGDClassifier`, `SGDClassifierLoss`, `SGDClassifierOptions`, `SGDRegressor`, `SGDRegressorOptions`, `SimpleImputer`, `SimpleImputerOptions`, `StackingClassifier`, `StackingClassifierOptions`, `StackingEstimatorSpec`, `StackingMethod`, `StackingRegressor`, `StackingRegressorEstimatorSpec`, `StackingRegressorOptions`, `StratifiedKFold`, `StratifiedKFoldOptions`, `StratifiedShuffleSplit`, `StratifiedShuffleSplitOptions`, `Transformer`, `TruncatedSVD`, `TruncatedSVDOptions`, `VarianceThreshold`, `VarianceThresholdOptions`, `VotingClassifier`, `VotingClassifierOptions`, `VotingEstimatorSpec`, `VotingRegressor`, `VotingRegressorEstimatorSpec`, `VotingRegressorOptions`, `VotingStrategy`.
