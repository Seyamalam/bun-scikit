import { expect, test } from "bun:test";
import { AgglomerativeClustering } from "../src";

test("AgglomerativeClustering separates two distant groups", () => {
  const X = [
    [0.0, 0.0],
    [0.2, -0.1],
    [-0.1, 0.2],
    [8.0, 8.0],
    [8.1, 8.2],
    [7.9, 8.0],
  ];

  const model = new AgglomerativeClustering({
    nClusters: 2,
    linkage: "ward",
  }).fit(X);

  expect(model.labels_).not.toBeNull();
  expect(model.children_).not.toBeNull();
  expect(model.distances_).not.toBeNull();
  expect(model.nMerges_).toBe(X.length - 1);
  expect(model.children_!.length).toBe(X.length - 1);
  expect(model.distances_!.length).toBe(X.length - 1);

  const labels = model.labels_!;
  expect(labels[0]).toBe(labels[1]);
  expect(labels[1]).toBe(labels[2]);
  expect(labels[3]).toBe(labels[4]);
  expect(labels[4]).toBe(labels[5]);
  expect(labels[0]).not.toBe(labels[3]);
});

test("AgglomerativeClustering supports complete linkage", () => {
  const X = [[0], [0.1], [5], [5.1]];
  const labels = new AgglomerativeClustering({
    nClusters: 2,
    linkage: "complete",
  }).fitPredict(X);
  expect(new Set(labels).size).toBe(2);
});

test("AgglomerativeClustering validates invalid nClusters", () => {
  expect(() => new AgglomerativeClustering({ nClusters: 0 })).toThrow(
    /nclusters must be an integer >= 1/i,
  );
  expect(() =>
    new AgglomerativeClustering({ nClusters: 3 }).fit([
      [0, 0],
      [1, 1],
    ]),
  ).toThrow(/cannot exceed sample count/i);
});
