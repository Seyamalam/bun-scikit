import { expect, test } from "bun:test";
import { KMeans } from "../src";

function sortCenters(centers: number[][]): number[][] {
  return centers.slice().sort((a, b) => a[0] - b[0]);
}

test("KMeans separates two well-defined clusters", () => {
  const X = [
    [0, 0],
    [0.1, -0.1],
    [-0.2, 0.1],
    [10, 10],
    [10.2, 9.9],
    [9.8, 10.1],
  ];

  const model = new KMeans({
    nClusters: 2,
    randomState: 42,
    nInit: 8,
    maxIter: 100,
  });
  model.fit(X);

  expect(model.clusterCenters_).not.toBeNull();
  expect(model.labels_).not.toBeNull();
  expect(model.inertia_).not.toBeNull();
  expect(model.nIter_).toBeGreaterThan(0);

  const centers = sortCenters(model.clusterCenters_!);
  expect(centers[0][0]).toBeCloseTo(-0.0333, 2);
  expect(centers[0][1]).toBeCloseTo(0.0, 1);
  expect(centers[1][0]).toBeCloseTo(10.0, 1);
  expect(centers[1][1]).toBeCloseTo(10.0, 1);

  const uniqueLabels = new Set(model.labels_!);
  expect(uniqueLabels.size).toBe(2);
});

test("KMeans is deterministic for fixed randomState", () => {
  const X = [
    [0, 0],
    [0.1, -0.1],
    [-0.2, 0.1],
    [10, 10],
    [10.2, 9.9],
    [9.8, 10.1],
    [5, 5],
    [5.1, 4.9],
    [4.8, 5.2],
  ];

  const a = new KMeans({ nClusters: 3, randomState: 7, nInit: 6 }).fit(X);
  const b = new KMeans({ nClusters: 3, randomState: 7, nInit: 6 }).fit(X);

  expect(a.labels_).toEqual(b.labels_);
  expect(a.inertia_).toBeCloseTo(b.inertia_!, 12);
  expect(a.clusterCenters_).toEqual(b.clusterCenters_);
});

test("KMeans exposes sklearn-style transform and score", () => {
  const X = [
    [0, 0],
    [1, 1],
    [9, 9],
    [10, 10],
  ];
  const model = new KMeans({ nClusters: 2, randomState: 123, nInit: 4 }).fit(X);

  const distances = model.transform(X);
  expect(distances.length).toBe(X.length);
  expect(distances[0].length).toBe(2);

  const labels = model.predict(X);
  expect(labels.length).toBe(X.length);

  const score = model.score(X);
  expect(score).toBeCloseTo(-model.inertia_!, 10);
  expect(score).toBeLessThan(0);
});

test("KMeans validates fitting preconditions", () => {
  const model = new KMeans({ nClusters: 2 });
  expect(() => model.predict([[0, 0]])).toThrow(/has not been fitted/i);
  expect(() => model.fit([[1, 1]])).toThrow(/cannot exceed sample count/i);
});
