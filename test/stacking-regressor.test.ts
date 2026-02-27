import { expect, test } from "bun:test";
import {
  DecisionTreeRegressor,
  DummyRegressor,
  StackingRegressor,
} from "../src";

test("StackingRegressor fits base estimators and final estimator", () => {
  const X = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]];
  const y = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23];

  const model = new StackingRegressor(
    [
      [
        "tree",
        () =>
          new DecisionTreeRegressor({
            maxDepth: 6,
            minSamplesLeaf: 1,
            randomState: 7,
          }),
      ],
      ["dummy", () => new DummyRegressor({ strategy: "mean" })],
    ],
    () =>
      new DecisionTreeRegressor({
        maxDepth: 6,
        minSamplesLeaf: 1,
        randomState: 11,
      }),
    { cv: 5, passthrough: true, randomState: 19 },
  ).fit(X, y);

  const preds = model.predict(X);
  expect(preds.length).toBe(X.length);
  expect(model.score(X, y)).toBeGreaterThan(0.85);
});

test("StackingRegressor enforces unique estimator names", () => {
  const X = [[0], [1], [2], [3], [4], [5]];
  const y = [0, 1, 4, 9, 16, 25];

  const model = new StackingRegressor(
    [
      ["dup", () => new DummyRegressor({ strategy: "mean" })],
      ["dup", () => new DummyRegressor({ strategy: "constant", constant: 3 })],
    ],
    () => new DummyRegressor({ strategy: "mean" }),
    { cv: 3 },
  );

  expect(() => model.fit(X, y)).toThrow(/must be unique/i);
});

test("StackingRegressor validates cv against sample size", () => {
  const X = [[0], [1], [2], [3]];
  const y = [0, 1, 4, 9];

  const model = new StackingRegressor(
    [["dummy", () => new DummyRegressor({ strategy: "mean" })]],
    () => new DummyRegressor({ strategy: "mean" }),
    { cv: 5 },
  );

  expect(() => model.fit(X, y)).toThrow(/cannot exceed sample count/i);
});
