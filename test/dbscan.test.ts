import { expect, test } from "bun:test";
import { DBSCAN } from "../src";

test("DBSCAN finds dense clusters and marks noise points", () => {
  const X = [
    [0.0, 0.0],
    [0.1, 0.0],
    [0.0, 0.1],
    [5.0, 5.0],
    [5.1, 5.0],
    [5.0, 5.1],
    [20.0, 20.0], // noise
  ];

  const model = new DBSCAN({ eps: 0.25, minSamples: 2 }).fit(X);

  expect(model.labels_).not.toBeNull();
  expect(model.coreSampleIndices_).not.toBeNull();
  expect(model.components_).not.toBeNull();
  expect(model.nClusters_).toBe(2);

  const labels = model.labels_!;
  expect(labels[6]).toBe(-1);
  const clusterSet = new Set(labels.filter((label) => label >= 0));
  expect(clusterSet.size).toBe(2);
});

test("DBSCAN fitPredict returns fitted labels", () => {
  const X = [
    [1.0, 1.0],
    [1.1, 1.0],
    [10.0, 10.0],
    [10.1, 10.0],
  ];
  const model = new DBSCAN({ eps: 0.2, minSamples: 2 });
  const labels = model.fitPredict(X);
  expect(model.labels_).not.toBeNull();
  expect(labels).toEqual(model.labels_!);
});

test("DBSCAN validates constructor options", () => {
  expect(() => new DBSCAN({ eps: 0 })).toThrow(/eps must be finite and > 0/i);
  expect(() => new DBSCAN({ minSamples: 0 })).toThrow(/minsamples must be an integer >= 1/i);
});
