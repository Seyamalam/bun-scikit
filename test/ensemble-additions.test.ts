import { expect, test } from "bun:test";
import { AdaBoostRegressor } from "../src/ensemble/AdaBoostRegressor";
import { ExtraTreesClassifier } from "../src/ensemble/ExtraTreesClassifier";
import { ExtraTreesRegressor } from "../src/ensemble/ExtraTreesRegressor";

test("AdaBoostRegressor fits a low-noise regression dataset", () => {
  const X = [[0], [1], [2], [3], [4], [5], [6], [7]];
  const y = [1, 3, 5, 7, 9, 11, 13, 15];
  const model = new AdaBoostRegressor(null, { nEstimators: 30, learningRate: 0.8, randomState: 42 }).fit(X, y);
  expect(model.score(X, y)).toBeGreaterThan(0.95);
  expect(model.predict([[8]])[0]).toBeGreaterThan(13);
});

test("ExtraTreesClassifier fits a separable dataset", () => {
  const X = [[-3], [-2], [-1], [1], [2], [3], [4], [5]];
  const y = [0, 0, 0, 1, 1, 1, 1, 1];
  const model = new ExtraTreesClassifier({ nEstimators: 40, randomState: 7, maxDepth: 4 }).fit(X, y);
  expect(model.score(X, y)).toBeGreaterThan(0.95);
});

test("ExtraTreesRegressor fits a smooth signal", () => {
  const X = [[0], [1], [2], [3], [4], [5], [6], [7]];
  const y = [0, 1, 4, 9, 16, 25, 36, 49];
  const model = new ExtraTreesRegressor({ nEstimators: 60, randomState: 11, maxDepth: 8 }).fit(X, y);
  expect(model.score(X, y)).toBeGreaterThan(0.9);
});
