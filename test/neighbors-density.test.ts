import { expect, test } from "bun:test";
import {
  KernelDensity,
  NearestNeighbors,
  RadiusNeighborsClassifier,
  RadiusNeighborsRegressor,
} from "../src";

test("NearestNeighbors exposes kneighbors and radiusNeighbors queries", () => {
  const X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [5, 5],
  ];
  const nn = new NearestNeighbors({ nNeighbors: 2, radius: 1.5 }).fit(X);
  const knn = nn.kneighbors([[0.1, 0.1]]);
  expect(knn.indices[0].length).toBe(2);

  const radius = nn.radiusNeighbors([[0.1, 0.1]]);
  expect(radius.indices[0].length).toBeGreaterThan(0);
});

test("RadiusNeighborsClassifier supports outlier labels", () => {
  const X = [[0], [0.2], [1], [1.2]];
  const y = [0, 0, 1, 1];
  const clf = new RadiusNeighborsClassifier({
    radius: 0.25,
    outlierLabel: -1,
  }).fit(X, y);
  const pred = clf.predict([[0.1], [1.1], [3.0]]);
  expect(pred).toEqual([0, 1, -1]);
});

test("RadiusNeighborsRegressor predicts NaN when no neighbors are in radius", () => {
  const X = [[0], [1], [2], [3]];
  const y = [0, 1, 4, 9];
  const reg = new RadiusNeighborsRegressor({ radius: 0.3 }).fit(X, y);
  const pred = reg.predict([[1.05], [10]]);
  expect(pred[0]).toBeFinite();
  expect(Number.isNaN(pred[1])).toBe(true);
});

test("KernelDensity estimates log density and can sample", () => {
  const X = [[0], [0.2], [0.1], [1.0], [1.2], [0.9]];
  const kde = new KernelDensity({ bandwidth: 0.2, kernel: "gaussian" }).fit(X);
  const scores = kde.scoreSamples([[0.1], [3]]);
  expect(scores[0]).toBeGreaterThan(scores[1]);
  const sample = kde.sample(3, 7);
  expect(sample.length).toBe(3);
  expect(sample[0].length).toBe(1);
});
