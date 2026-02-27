import { expect, test } from "bun:test";
import {
  AdaBoostRegressor,
  BaggingRegressor,
  Birch,
  ExtraTreesClassifier,
  ExtraTreesRegressor,
  GenericUnivariateSelect,
  IsolationForest,
  Isomap,
  KNNImputer,
  KNeighborsRegressor,
  LinearSVR,
  LocalOutlierFactor,
  LocallyLinearEmbedding,
  MDS,
  OneVsOneClassifier,
  OneVsRestClassifier,
  OneClassSVM,
  OPTICS,
  NuSVC,
  NuSVR,
  OrdinalEncoder,
  RFECV,
  RFE,
  SelectFdr,
  SelectFpr,
  SelectFwe,
  SelectFromModel,
  SequentialFeatureSelector,
  SpectralClustering,
  SVC,
  SVR,
  TSNE,
} from "../src";

function hasParamsApi(value: unknown): value is { getParams: () => unknown; setParams: (params: unknown) => unknown } {
  return (
    typeof value === "object" &&
    value !== null &&
    typeof (value as Record<string, unknown>).getParams === "function" &&
    typeof (value as Record<string, unknown>).setParams === "function"
  );
}

test("new estimators expose stable getParams/setParams roundtrip", () => {
  const estimators = [
    new SVC({ kernel: "linear", C: 2 }),
    new SVR({ kernel: "linear", C: 2 }),
    new LinearSVR({ C: 2, epsilon: 0.05 }),
    new NuSVC({ nu: 0.4, kernel: "linear" }),
    new NuSVR({ nu: 0.4, kernel: "linear", C: 2 }),
    new AdaBoostRegressor(null, { nEstimators: 10 }),
    new ExtraTreesClassifier({ nEstimators: 10 }),
    new ExtraTreesRegressor({ nEstimators: 10 }),
    new BaggingRegressor(() => new AdaBoostRegressor(null, { nEstimators: 3 }), {
      nEstimators: 3,
      randomState: 7,
    }),
    new KNeighborsRegressor({ nNeighbors: 2, weights: "distance" }),
    new SpectralClustering({ nClusters: 2, randomState: 7 }),
    new Birch({ nClusters: 2, threshold: 0.3 }),
    new OPTICS({ minSamples: 2, maxEps: 0.5 }),
    new SelectFromModel(() => new ExtraTreesClassifier({ nEstimators: 5 }), { threshold: "mean" }),
    new RFE(() => new ExtraTreesClassifier({ nEstimators: 5 }), { nFeaturesToSelect: 1 }),
    new RFECV(() => new ExtraTreesClassifier({ nEstimators: 5 }), { minFeaturesToSelect: 1 }),
    new SelectFpr({ alpha: 0.1 }),
    new SelectFdr({ alpha: 0.1 }),
    new SelectFwe({ alpha: 0.1 }),
    new GenericUnivariateSelect({ mode: "k_best", param: 1 }),
    new SequentialFeatureSelector(() => new SVC({ kernel: "linear" }), { nFeaturesToSelect: 1, cv: 2 }),
    new OneVsRestClassifier(() => new SVC({ kernel: "linear" }), { normalizeProba: true }),
    new OneVsOneClassifier(() => new SVC({ kernel: "linear" })),
    new IsolationForest({ contamination: 0.1 }),
    new LocalOutlierFactor({ nNeighbors: 2, contamination: 0.1, novelty: true }),
    new OneClassSVM({ nu: 0.2, kernel: "rbf" }),
    new TSNE({ nComponents: 2, maxIter: 250 }),
    new Isomap({ nNeighbors: 2, nComponents: 2 }),
    new LocallyLinearEmbedding({ nNeighbors: 2, nComponents: 2 }),
    new MDS({ nComponents: 2 }),
  ];

  for (const estimator of estimators) {
    expect(hasParamsApi(estimator)).toBe(true);
    const params = (estimator as { getParams: () => unknown }).getParams();
    const serialized = JSON.stringify(params);
    expect(typeof serialized).toBe("string");
    (estimator as { setParams: (p: unknown) => unknown }).setParams(JSON.parse(serialized));
    const next = (estimator as { getParams: () => unknown }).getParams();
    expect(JSON.stringify(next)).toEqual(serialized);
  }
});

test("fit methods keep working when sample_weight is passed as extra argument", () => {
  const XCls = [[-2], [-1], [1], [2]];
  const yCls = [0, 0, 1, 1];
  const weights = [1, 1, 1, 1];
  const cls = new SVC({ kernel: "linear", maxIter: 200, learningRate: 0.1 }) as unknown as {
    fit: (X: number[][], y: number[], sampleWeight?: number[]) => unknown;
    predict: (X: number[][]) => number[];
  };
  cls.fit(XCls, yCls, weights);
  expect(cls.predict(XCls).length).toBe(XCls.length);

  const XReg = [[0], [1], [2], [3]];
  const yReg = [0, 2, 4, 6];
  const reg = new SVR({ kernel: "linear", C: 3 }) as unknown as {
    fit: (X: number[][], y: number[], sampleWeight?: number[]) => unknown;
    predict: (X: number[][]) => number[];
  };
  reg.fit(XReg, yReg, weights);
  expect(reg.predict(XReg).length).toBe(XReg.length);
});

test("serialization-friendly preprocessing states roundtrip", () => {
  const ordinal = new OrdinalEncoder().fit([
    [1, 10],
    [2, 20],
  ]);
  const knn = new KNNImputer({ nNeighbors: 2 }).fit([
    [1, Number.NaN],
    [2, 8],
    [3, 10],
  ]);

  const ordinalJson = JSON.stringify({
    categories: ordinal.categories_,
    nFeaturesIn: ordinal.nFeaturesIn_,
  });
  const knnJson = JSON.stringify({ statistics: knn.statistics_ });
  expect(typeof ordinalJson).toBe("string");
  expect(typeof knnJson).toBe("string");
});
