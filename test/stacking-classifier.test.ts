import { expect, test } from "bun:test";
import {
  DummyClassifier,
  GaussianNB,
  KNeighborsClassifier,
  StackingClassifier,
} from "../src";

test("StackingClassifier fits base estimators and final estimator", () => {
  const X = [
    [0.0, 0.0],
    [0.2, 0.1],
    [0.3, -0.1],
    [1.0, 1.0],
    [1.2, 1.1],
    [1.1, 0.9],
    [3.0, 3.0],
    [3.1, 3.0],
    [2.9, 3.2],
    [4.0, 4.0],
    [4.1, 3.9],
    [3.9, 4.1],
  ];
  const y = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];

  const model = new StackingClassifier(
    [
      ["gnb", () => new GaussianNB()],
      ["dummy", () => new DummyClassifier({ strategy: "prior" })],
    ],
    () => new GaussianNB(),
    { cv: 3, passthrough: true, stackMethod: "auto", randomState: 11 },
  ).fit(X, y);

  const preds = model.predict(X);
  expect(preds.length).toBe(X.length);
  expect(model.score(X, y)).toBeGreaterThan(0.8);

  const proba = model.predictProba(X);
  expect(proba.length).toBe(X.length);
  expect(proba[0].length).toBe(2);
});

test("StackingClassifier validates stackMethod predictProba", () => {
  const X = [[0], [1], [2], [3], [4], [5]];
  const y = [0, 0, 0, 1, 1, 1];
  const model = new StackingClassifier(
    [["knn", () => new KNeighborsClassifier({ nNeighbors: 1 })]],
    () => new GaussianNB(),
    { cv: 3, stackMethod: "predictProba" },
  );

  expect(() => model.fit(X, y)).toThrow(/requires base estimators with predictproba/i);
});

test("StackingClassifier enforces unique base estimator names", () => {
  const X = [[0], [1], [2], [3], [4], [5]];
  const y = [0, 0, 0, 1, 1, 1];
  const model = new StackingClassifier(
    [
      ["dup", () => new GaussianNB()],
      ["dup", () => new DummyClassifier({ strategy: "prior" })],
    ],
    () => new GaussianNB(),
    { cv: 3 },
  );
  expect(() => model.fit(X, y)).toThrow(/must be unique/i);
});
