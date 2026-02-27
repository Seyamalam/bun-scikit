import { expect, test } from "bun:test";
import { OPTICS } from "../src";

test("OPTICS computes ordering/reachability and extracts clusters", () => {
  const X = [
    [0.0, 0.0],
    [0.1, 0.0],
    [0.0, 0.1],
    [5.0, 5.0],
    [5.1, 5.0],
    [5.0, 5.1],
    [20.0, 20.0],
  ];

  const model = new OPTICS({
    minSamples: 2,
    maxEps: 0.25,
    clusterMethod: "dbscan",
  }).fit(X);

  expect(model.labels_).not.toBeNull();
  expect(model.ordering_).not.toBeNull();
  expect(model.reachability_).not.toBeNull();
  expect(model.coreDistances_).not.toBeNull();
  expect(model.predecessor_).not.toBeNull();
  expect(model.ordering_!.length).toBe(X.length);

  const labels = model.labels_!;
  expect(labels[6]).toBe(-1);
  const clusterSet = new Set(labels.filter((label) => label >= 0));
  expect(clusterSet.size).toBe(2);
});
