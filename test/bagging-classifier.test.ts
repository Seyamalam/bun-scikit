import { expect, test } from "bun:test";
import { BaggingClassifier, GaussianNB } from "../src";

test("BaggingClassifier fits bootstrap ensembles and predicts probabilities", () => {
  const X = [
    [0.0, 0.1],
    [0.1, 0.2],
    [0.2, 0.0],
    [0.3, 0.2],
    [1.0, 1.1],
    [1.1, 0.9],
    [0.9, 1.2],
    [1.2, 1.0],
  ];
  const y = [0, 0, 0, 0, 1, 1, 1, 1];

  const model = new BaggingClassifier(() => new GaussianNB(), {
    nEstimators: 15,
    maxSamples: 0.75,
    maxFeatures: 1.0,
    bootstrap: true,
    randomState: 21,
  }).fit(X, y);

  const preds = model.predict(X);
  expect(preds.length).toBe(X.length);
  expect(model.score(X, y)).toBeGreaterThan(0.85);

  const proba = model.predictProba(X);
  expect(proba.length).toBe(X.length);
  expect(proba[0].length).toBe(2);
  for (let i = 0; i < proba.length; i += 1) {
    expect(proba[i][0] + proba[i][1]).toBeCloseTo(1, 10);
  }
});

test("BaggingClassifier is deterministic with fixed randomState", () => {
  const X = [[0], [1], [2], [3], [4], [5], [6], [7]];
  const y = [0, 0, 0, 0, 1, 1, 1, 1];

  const a = new BaggingClassifier(() => new GaussianNB(), {
    nEstimators: 11,
    maxSamples: 0.8,
    randomState: 9,
  }).fit(X, y);
  const b = new BaggingClassifier(() => new GaussianNB(), {
    nEstimators: 11,
    maxSamples: 0.8,
    randomState: 9,
  }).fit(X, y);

  expect(a.predict(X)).toEqual(b.predict(X));
  expect(a.predictProba(X)).toEqual(b.predictProba(X));
});

test("BaggingClassifier validates fit and feature dimensions", () => {
  const model = new BaggingClassifier(() => new GaussianNB());
  expect(() => model.predict([[0, 1]])).toThrow(/has not been fitted/i);

  model.fit(
    [
      [0, 0],
      [1, 1],
      [0, 1],
      [1, 0],
    ],
    [0, 1, 0, 1],
  );

  expect(() => model.predict([[0, 1, 2]])).toThrow(/feature size mismatch/i);
});
