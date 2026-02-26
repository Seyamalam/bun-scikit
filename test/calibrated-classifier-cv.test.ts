import { expect, test } from "bun:test";
import { CalibratedClassifierCV, GaussianNB } from "../src";

class LabelOnlyThresholdEstimator {
  private threshold = 0;

  fit(X: number[][]): this {
    let mean = 0;
    for (let i = 0; i < X.length; i += 1) {
      mean += X[i][0];
    }
    this.threshold = mean / X.length;
    return this;
  }

  predict(X: number[][]): number[] {
    return X.map((row) => (row[0] >= this.threshold ? 1 : 0));
  }
}

test("CalibratedClassifierCV fits sigmoid calibration in non-ensemble mode", () => {
  const X = [[0], [0.2], [0.4], [0.6], [0.8], [1.0], [1.2], [1.4], [1.6], [1.8]];
  const y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1];

  const model = new CalibratedClassifierCV(() => new GaussianNB(), {
    cv: 4,
    method: "sigmoid",
    ensemble: false,
    randomState: 9,
  }).fit(X, y);

  const proba = model.predictProba(X);
  expect(proba.length).toBe(X.length);
  expect(proba[0].length).toBe(2);
  for (let i = 0; i < proba.length; i += 1) {
    expect(proba[i][0]).toBeGreaterThanOrEqual(0);
    expect(proba[i][1]).toBeLessThanOrEqual(1);
    expect(proba[i][0] + proba[i][1]).toBeCloseTo(1, 10);
  }
});

test("CalibratedClassifierCV supports isotonic calibration and estimator fallback", () => {
  const X = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]];
  const y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1];

  const model = new CalibratedClassifierCV(() => new LabelOnlyThresholdEstimator(), {
    cv: 4,
    method: "isotonic",
    ensemble: true,
    randomState: 17,
  }).fit(X, y);

  const preds = model.predict(X);
  expect(preds.length).toBe(X.length);
  expect(model.score(X, y)).toBeGreaterThan(0.8);
});

test("CalibratedClassifierCV throws before fit", () => {
  const model = new CalibratedClassifierCV(() => new GaussianNB());
  expect(() => model.predict([[0]])).toThrow(/has not been fitted/i);
});
