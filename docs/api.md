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
- `featureImportances_: number[] | null`
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
- `featureImportances_: number[] | null`
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
- `featureImportances_: number[] | null`

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
- `featureImportances_: number[] | null`

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
- `featureImportances_: number[] | null`

### `HistGradientBoostingClassifierOptions`

- `maxIter?: number`
- `learningRate?: number`
- `maxBins?: number`
- `maxDepth?: number`
- `maxLeafNodes?: number`
- `minSamplesLeaf?: number`
- `l2Regularization?: number`
- `earlyStopping?: boolean`
- `nIterNoChange?: number`
- `validationFraction?: number`
- `tolerance?: number`
- `randomState?: number`

### `HistGradientBoostingClassifier`

Methods:

- `fit(X: number[][], y: number[]): this`
- `decisionFunction(X: number[][]): number[]`
- `predict(X: number[][]): number[]`
- `predictProba(X: number[][]): number[][]`
- `score(X: number[][], y: number[]): number`

Learned attributes:

- `classes_: number[]`
- `estimators_: unknown[]`
- `baselineLogit_: number | null`
- `featureImportances_: number[] | null`

### `HistGradientBoostingRegressorOptions`

- `maxIter?: number`
- `learningRate?: number`
- `maxBins?: number`
- `maxDepth?: number`
- `maxLeafNodes?: number`
- `minSamplesLeaf?: number`
- `l2Regularization?: number`
- `earlyStopping?: boolean`
- `nIterNoChange?: number`
- `validationFraction?: number`
- `tolerance?: number`
- `randomState?: number`

### `HistGradientBoostingRegressor`

Methods:

- `fit(X: number[][], y: number[]): this`
- `predict(X: number[][]): number[]`
- `score(X: number[][], y: number[]): number`

Learned attributes:

- `estimators_: unknown[]`
- `baseline_: number | null`
- `featureImportances_: number[] | null`

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

### `KernelPCAKernel`

- `"linear" | "rbf" | "poly"`

### `KernelPCAOptions`

- `nComponents?: number`
- `kernel?: KernelPCAKernel`
- `gamma?: number`
- `degree?: number`
- `coef0?: number`
- `tolerance?: number`
- `maxIter?: number`

### `KernelPCA`

Methods:

- `fit(X: number[][]): this`
- `transform(X: number[][]): number[][]`
- `fitTransform(X: number[][]): number[][]`

Learned attributes:

- `alphas_: number[][] | null`
- `lambdas_: number[] | null`
- `nFeaturesIn_: number | null`

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

## Recent Additions

- `AdaBoostRegressor`
- `AdaBoostRegressorOptions`
- `crossValidate`
- `CrossValidateOptions`
- `CrossValidateResult`
- `ExtraTreesClassifier`
- `ExtraTreesClassifierOptions`
- `ExtraTreesRegressor`
- `ExtraTreesRegressorOptions`
- `GroupKFold`
- `GroupKFoldOptions`
- `GroupShuffleSplit`
- `GroupShuffleSplitOptions`
- `KNNImputer`
- `KNNImputerOptions`
- `LinearSVR`
- `LinearSVROptions`
- `NuSVC`
- `NuSVCOptions`
- `NuSVR`
- `NuSVROptions`
- `OrdinalEncoder`
- `OrdinalEncoderOptions`
- `SVC`
- `SVCOptions`
- `SVR`
- `SVROptions`
- `StratifiedGroupKFold`
- `StratifiedGroupKFoldOptions`
- `permutationImportance`
- `PermutationImportanceEstimator`
- `PermutationImportanceOptions`
- `PermutationImportanceResult`

- `chi2`
- `f_classif`
- `f_regression`
- `FRegressionOptions`
- `SelectKBest`
- `SelectKBestOptions`
- `SelectPercentile`
- `SelectPercentileOptions`
- `UnivariateScoreFunc`
- `UnivariateScoreResult`
- `BaggingRegressor`
- `BaggingRegressorOptions`
- `Birch`
- `BirchOptions`
- `KNeighborsRegressor`
- `KNeighborsRegressorOptions`
- `KNeighborsRegressorWeights`
- `mutualInfoClassif`
- `MutualInfoOptions`
- `mutualInfoRegression`
- `OneVsOneClassifier`
- `OneVsRestClassifier`
- `OneVsRestClassifierOptions`
- `OPTICS`
- `OPTICSClusterMethod`
- `OPTICSOptions`
- `partialDependence`
- `PartialDependenceEstimator`
- `PartialDependenceOptions`
- `PartialDependenceResponseMethod`
- `PartialDependenceResult`
- `RFE`
- `RFECV`
- `RFECVOptions`
- `RFEOptions`
- `SelectFromModel`
- `SelectFromModelOptions`
- `SpectralAffinity`
- `SpectralClustering`
- `SpectralClusteringOptions`
- `adjustedRandScore`
- `calinskiHarabaszScore`
- `daviesBouldinScore`
- `GenericUnivariateSelect`
- `GenericUnivariateSelectMode`
- `GenericUnivariateSelectOptions`
- `IsolationForest`
- `IsolationForestContamination`
- `IsolationForestOptions`
- `Isomap`
- `IsomapOptions`
- `LocallyLinearEmbedding`
- `LocallyLinearEmbeddingOptions`
- `LocalOutlierFactor`
- `LocalOutlierFactorContamination`
- `LocalOutlierFactorOptions`
- `MDS`
- `MDSDissimilarity`
- `MDSOptions`
- `OneClassSVM`
- `OneClassSVMKernel`
- `OneClassSVMOptions`
- `PartialDependenceFeature`
- `permutationTestScore`
- `PermutationTestScoreOptions`
- `PermutationTestScoreResult`
- `SelectFdr`
- `SelectFdrOptions`
- `SelectFpr`
- `SelectFprOptions`
- `SelectFwe`
- `SelectFweOptions`
- `SequentialFeatureSelector`
- `SequentialFeatureSelectorDirection`
- `SequentialFeatureSelectorOptions`
- `silhouetteScore`
- `TSNE`
- `TSNEOptions`
- `PLSSVD`
- `PLSSVDOptions`
- `PLSRegression`
- `PLSRegressionOptions`
- `PLSCanonical`
- `PLSCanonicalOptions`
- `CCA`
- `CCAOptions`
- `SparsePCA`
- `SparsePCAOptions`
- `MiniBatchSparsePCA`
- `MiniBatchSparsePCAOptions`
- `DictionaryLearning`
- `DictionaryLearningOptions`
- `MiniBatchDictionaryLearning`
- `MiniBatchDictionaryLearningOptions`
- `EmpiricalCovariance`
- `EmpiricalCovarianceOptions`
- `LedoitWolf`
- `OAS`
- `MinCovDet`
- `MinCovDetOptions`
- `NearestNeighbors`
- `NearestNeighborsOptions`
- `NeighborQueryResult`
- `RadiusNeighborsClassifier`
- `RadiusNeighborsClassifierOptions`
- `RadiusNeighborsClassifierWeights`
- `RadiusNeighborsRegressor`
- `RadiusNeighborsRegressorOptions`
- `RadiusNeighborsRegressorWeights`
- `KernelDensity`
- `KernelDensityKernel`
- `KernelDensityOptions`
- `LabelPropagation`
- `LabelPropagationOptions`
- `LabelSpreading`
- `LabelSpreadingOptions`
- `MLPClassifier`
- `MLPClassifierOptions`
- `MLPRegressor`
- `MLPRegressorOptions`
- `Ridge`
- `RidgeOptions`
- `Lasso`
- `LassoOptions`
- `ElasticNet`
- `ElasticNetOptions`
- `RidgeCV`
- `RidgeCVOptions`
- `LassoCV`
- `LassoCVOptions`
- `ElasticNetCV`
- `ElasticNetCVOptions`
- `LinearDiscriminantAnalysis`
- `LinearDiscriminantAnalysisOptions`
- `QuadraticDiscriminantAnalysis`
- `QuadraticDiscriminantAnalysisOptions`
- `GaussianMixture`
- `GaussianMixtureOptions`
- `BayesianGaussianMixture`
- `BayesianGaussianMixtureOptions`
- `QuantileTransformer`
- `QuantileTransformerOptions`
- `QuantileOutputDistribution`
- `PowerTransformer`
- `PowerTransformerOptions`
- `PowerTransformerMethod`
- `KBinsDiscretizer`
- `KBinsDiscretizerOptions`
- `KBinsEncode`
- `KBinsStrategy`
- `IterativeImputer`
- `IterativeImputerOptions`
- `IterativeImputerInitialStrategy`
- `MissingIndicator`
- `MissingIndicatorOptions`
- `MissingIndicatorFeatures`
- `MiniBatchKMeans`
- `MiniBatchKMeansOptions`
- `MeanShift`
- `MeanShiftOptions`
- `AffinityPropagation`
- `AffinityPropagationOptions`
- `IncrementalPCA`
- `IncrementalPCAOptions`
- `FactorAnalysis`
- `FactorAnalysisOptions`
- `MiniBatchNMF`
- `MiniBatchNMFOptions`
- `ParameterGrid`
- `ParamGridInput`
- `expandParamGrid`
- `ParameterSampler`
- `drawParameterSamples`
- `MultiOutputClassifier`
- `MultiOutputRegressor`
- `ClassifierChain`
- `ClassifierChainOptions`
- `RegressorChain`
- `RegressorChainOptions`
## API Symbol Index

`AdaBoostClassifier`, `AdaBoostClassifierOptions`, `AgglomerativeClustering`, `AgglomerativeClusteringOptions`, `AgglomerativeLinkage`, `AgglomerativeMetric`, `BaggingClassifier`, `BaggingClassifierOptions`, `balancedAccuracyScore`, `Binarizer`, `BinarizerOptions`, `brierScoreLoss`, `BuiltInScoring`, `CalibratedClassifierCV`, `CalibratedClassifierCVOptions`, `CalibrationMethod`, `chi2`, `classificationReport`, `ClassificationReportLabelMetrics`, `ClassificationReportResult`, `ColumnSelector`, `ColumnTransformer`, `ColumnTransformerOptions`, `ColumnTransformerSpec`, `ColumnTransformerStep`, `confusionMatrix`, `ConfusionMatrixResult`, `CrossValEstimator`, `crossValPredict`, `CrossValPredictMethod`, `CrossValPredictOptions`, `crossValScore`, `CrossValScoreOptions`, `CrossValSplitter`, `DBSCAN`, `DBSCANOptions`, `DecisionTreeRegressor`, `DecisionTreeRegressorOptions`, `DummyClassifier`, `DummyClassifierOptions`, `DummyClassifierStrategy`, `DummyRegressor`, `DummyRegressorOptions`, `DummyRegressorStrategy`, `explainedVarianceScore`, `f_classif`, `f_regression`, `FastICA`, `FastICAOptions`, `FeatureUnion`, `FeatureUnionOptions`, `FeatureUnionSpec`, `FeatureUnionTransformer`, `FoldIndices`, `FRegressionOptions`, `GaussianNB`, `GaussianNBOptions`, `GradientBoostingClassifier`, `GradientBoostingClassifierOptions`, `GradientBoostingRegressor`, `GradientBoostingRegressorOptions`, `GridSearchCV`, `GridSearchCVOptions`, `GridSearchResultRow`, `GroupShuffleSplit`, `GroupShuffleSplitOptions`, `ImputerStrategy`, `KFold`, `KFoldOptions`, `KMeans`, `KMeansOptions`, `LabelEncoder`, `learningCurve`, `LearningCurveOptions`, `LearningCurveResult`, `LinearSVC`, `LinearSVCOptions`, `logLoss`, `matthewsCorrcoef`, `MaxAbsScaler`, `meanAbsolutePercentageError`, `MinMaxScaler`, `MinMaxScalerOptions`, `MultioutputMode`, `NMF`, `NMFOptions`, `Normalizer`, `NormalizerOptions`, `OneHotEncoder`, `OneHotEncoderOptions`, `ParamDistributions`, `ParamGrid`, `PCA`, `PCAOptions`, `permutationImportance`, `PermutationImportanceEstimator`, `PermutationImportanceOptions`, `PermutationImportanceResult`, `Pipeline`, `PipelineStep`, `PolynomialFeatures`, `PolynomialFeaturesOptions`, `RandomForestRegressor`, `RandomForestRegressorOptions`, `RandomizedSearchCV`, `RandomizedSearchCVOptions`, `RandomizedSearchResultRow`, `RegressionMetricOptions`, `RepeatedKFold`, `RepeatedKFoldOptions`, `RepeatedStratifiedKFold`, `RepeatedStratifiedKFoldOptions`, `RobustScaler`, `RobustScalerOptions`, `rocAucScore`, `ScoringFn`, `SelectKBest`, `SelectKBestOptions`, `SelectPercentile`, `SelectPercentileOptions`, `SGDClassifier`, `SGDClassifierLoss`, `SGDClassifierOptions`, `SGDRegressor`, `SGDRegressorOptions`, `SimpleImputer`, `SimpleImputerOptions`, `StackingClassifier`, `StackingClassifierOptions`, `StackingEstimatorSpec`, `StackingMethod`, `StackingRegressor`, `StackingRegressorEstimatorSpec`, `StackingRegressorOptions`, `StratifiedGroupKFold`, `StratifiedGroupKFoldOptions`, `StratifiedKFold`, `StratifiedKFoldOptions`, `StratifiedShuffleSplit`, `StratifiedShuffleSplitOptions`, `Transformer`, `TruncatedSVD`, `TruncatedSVDOptions`, `UnivariateScoreFunc`, `UnivariateScoreResult`, `validationCurve`, `ValidationCurveOptions`, `ValidationCurveResult`, `VarianceThreshold`, `VarianceThresholdOptions`, `VotingClassifier`, `VotingClassifierOptions`, `VotingEstimatorSpec`, `VotingRegressor`, `VotingRegressorEstimatorSpec`, `VotingRegressorOptions`, `VotingStrategy`.

## Parity Batch API Additions

`ARDRegression`, `ARDRegressionOptions`, `BallTree`, `BallTreeOptions`, `BayesianRidge`, `BayesianRidgeOptions`, `BernoulliNB`, `BernoulliNBOptions`, `CategoricalNB`, `CategoricalNBOptions`, `ComplementNB`, `ComplementNBOptions`, `DictSample`, `DictValue`, `DictVectorizer`, `DictVectorizerOptions`, `DistanceMetric`, `DistanceMetricName`, `DistanceMetricOptions`, `EllipticEnvelope`, `EllipticEnvelopeOptions`, `FeatureHasher`, `FeatureHasherInputType`, `FeatureHasherOptions`, `FunctionTransformer`, `FunctionTransformerOptions`, `GammaRegressor`, `GammaRegressorOptions`, `GaussianProcessClassifier`, `GaussianProcessClassifierOptions`, `GaussianProcessRegressor`, `GaussianProcessRegressorOptions`, `GraphicalLasso`, `GraphicalLassoCV`, `GraphicalLassoCVOptions`, `GraphicalLassoOptions`, `HashedDictSample`, `HashedPairSample`, `HashedStringSample`, `HuberRegressor`, `HuberRegressorOptions`, `IsotonicIncreasing`, `IsotonicOutOfBounds`, `IsotonicRegression`, `IsotonicRegressionOptions`, `KDTree`, `KDTreeOptions`, `KDTreeQueryRadiusResult`, `KDTreeQueryResult`, `KernelCenterer`, `KNeighborsTransformer`, `KNeighborsTransformerMode`, `KNeighborsTransformerOptions`, `LabelBinarizer`, `LabelBinarizerOptions`, `LogisticRegressionCV`, `LogisticRegressionCVOptions`, `MultiLabel`, `MultiLabelBinarizer`, `MultiLabelBinarizerOptions`, `MultinomialNB`, `MultinomialNBOptions`, `MultiTaskElasticNet`, `MultiTaskElasticNetCV`, `MultiTaskElasticNetCVOptions`, `MultiTaskElasticNetOptions`, `MultiTaskLasso`, `MultiTaskLassoCV`, `MultiTaskLassoCVOptions`, `MultiTaskLassoOptions`, `NearestCentroid`, `NearestCentroidMetric`, `NearestCentroidOptions`, `NeighborhoodComponentsAnalysis`, `NeighborhoodComponentsAnalysisOptions`, `PassiveAggressiveClassifier`, `PassiveAggressiveClassifierOptions`, `PassiveAggressiveLoss`, `PassiveAggressiveRegressor`, `PassiveAggressiveRegressorLoss`, `PassiveAggressiveRegressorOptions`, `Perceptron`, `PerceptronOptions`, `PoissonRegressor`, `PoissonRegressorOptions`, `QuantileRegressor`, `QuantileRegressorOptions`, `QueryRadiusResult`, `TransformFunction`, `TreeQueryResult`.
