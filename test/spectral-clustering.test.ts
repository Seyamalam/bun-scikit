import { expect, test } from "bun:test";
import { SpectralClustering } from "../src";

test("SpectralClustering separates two distant groups", () => {
  const X = [
    [0.0, 0.1],
    [0.1, 0.0],
    [-0.1, 0.0],
    [8.0, 8.0],
    [8.1, 8.2],
    [7.9, 8.1],
  ];
  const model = new SpectralClustering({
    nClusters: 2,
    affinity: "rbf",
    gamma: 0.5,
    randomState: 42,
  }).fit(X);

  expect(model.labels_).not.toBeNull();
  const labels = model.labels_!;
  expect(labels[0]).toBe(labels[1]);
  expect(labels[1]).toBe(labels[2]);
  expect(labels[3]).toBe(labels[4]);
  expect(labels[4]).toBe(labels[5]);
  expect(labels[0]).not.toBe(labels[3]);
});

test("SpectralClustering supports nearest-neighbors affinity", () => {
  const X = [[0], [0.1], [0.2], [4], [4.1], [4.2]];
  const labels = new SpectralClustering({
    nClusters: 2,
    affinity: "nearest_neighbors",
    nNeighbors: 2,
    randomState: 0,
  }).fitPredict(X);
  expect(new Set(labels).size).toBe(2);
});
