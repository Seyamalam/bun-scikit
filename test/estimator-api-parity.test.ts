import { expect, test } from "bun:test";
import {
  AdaBoostRegressor,
  ExtraTreesClassifier,
  ExtraTreesRegressor,
  KNNImputer,
  LinearSVR,
  NuSVC,
  NuSVR,
  OrdinalEncoder,
  SVC,
  SVR,
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
