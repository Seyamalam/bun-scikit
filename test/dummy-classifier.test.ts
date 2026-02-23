import { expect, test } from "bun:test";
import { DummyClassifier } from "../src/dummy/DummyClassifier";

test("DummyClassifier prior strategy predicts majority class", () => {
  const X = [[1], [2], [3], [4], [5], [6]];
  const y = [0, 0, 0, 1, 0, 1];

  const model = new DummyClassifier({ strategy: "prior" }).fit(X, y);
  expect(model.classes_).toEqual([0, 1]);
  expect(model.classPrior_![0]).toBeCloseTo(4 / 6, 10);
  expect(model.predict([[10], [11], [12]])).toEqual([0, 0, 0]);
});

test("DummyClassifier uniform strategy is deterministic with randomState", () => {
  const X = [[1], [2], [3], [4]];
  const y = [0, 1, 0, 1];

  const a = new DummyClassifier({ strategy: "uniform", randomState: 7 }).fit(X, y);
  const b = new DummyClassifier({ strategy: "uniform", randomState: 7 }).fit(X, y);

  expect(a.predict([[10], [20], [30], [40]])).toEqual(b.predict([[10], [20], [30], [40]]));
});

test("DummyClassifier constant strategy predicts configured label", () => {
  const X = [[1], [2], [3], [4]];
  const y = [0, 1, 1, 0];

  const model = new DummyClassifier({ strategy: "constant", constant: 1 }).fit(X, y);
  expect(model.predict([[10], [20], [30]])).toEqual([1, 1, 1]);
  expect(model.predictProba([[10]])).toEqual([[0, 1]]);
});
