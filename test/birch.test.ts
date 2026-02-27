import { expect, test } from "bun:test";
import { Birch } from "../src";

test("Birch finds compact clusters and predicts labels", () => {
  const X = [
    [0.0, 0.0],
    [0.1, 0.1],
    [-0.1, 0.0],
    [5.0, 5.0],
    [5.1, 5.0],
    [4.9, 5.2],
  ];
  const model = new Birch({
    threshold: 0.4,
    nClusters: 2,
    computeLabels: true,
  }).fit(X);

  expect(model.labels_).not.toBeNull();
  expect(model.subclusterCenters_).not.toBeNull();
  expect(model.clusterCenters_).not.toBeNull();

  const labels = model.labels_!;
  expect(labels[0]).toBe(labels[1]);
  expect(labels[1]).toBe(labels[2]);
  expect(labels[3]).toBe(labels[4]);
  expect(labels[4]).toBe(labels[5]);
  expect(labels[0]).not.toBe(labels[3]);

  const pred = model.predict([[0.05, 0.0], [5.05, 5.1]]);
  expect(pred[0]).not.toBe(pred[1]);
});
