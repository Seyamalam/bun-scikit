import { expect, test } from "bun:test";
import {
  BallTree,
  DistanceMetric,
  KDTree,
  KNeighborsTransformer,
  NearestCentroid,
  NeighborhoodComponentsAnalysis,
} from "../src";

const X = [
  [0, 0],
  [0.1, 0.2],
  [1, 1],
  [1.2, 0.9],
  [5, 5],
  [5.1, 5.2],
];

test("DistanceMetric pairwise distances are computed", () => {
  const metric = new DistanceMetric("euclidean");
  const matrix = metric.pairwise(X.slice(0, 2), X.slice(2, 4));
  expect(matrix.length).toBe(2);
  expect(matrix[0].length).toBe(2);
});

test("BallTree and KDTree query and queryRadius", () => {
  const ball = new BallTree(X);
  const kd = new KDTree(X);

  const ballQuery = ball.query([[0, 0]], 2);
  const kdQuery = kd.query([[0, 0]], 2);
  expect(ballQuery.indices[0].length).toBe(2);
  expect(kdQuery.indices[0].length).toBe(2);

  const ballRadius = ball.queryRadius([[0, 0]], 0.5, false, true) as { indices: number[][]; distances?: number[][] };
  const kdRadius = kd.queryRadius([[0, 0]], 0.5, false, true) as { indices: number[][]; distances?: number[][] };
  expect(ballRadius.indices.length).toBe(1);
  expect(kdRadius.indices.length).toBe(1);
});

test("KNeighborsTransformer, NCA, and NearestCentroid work on simple data", () => {
  const y = [0, 0, 0, 0, 1, 1];

  const transformer = new KNeighborsTransformer({ nNeighbors: 2, mode: "distance" }).fit(X);
  const graph = transformer.transform(X);
  expect(graph.length).toBe(X.length);
  expect(graph[0].length).toBe(X.length);

  const nca = new NeighborhoodComponentsAnalysis({ nComponents: 1, maxIter: 40, randomState: 7 }).fit(X, y);
  const embedded = nca.transform(X);
  expect(embedded[0].length).toBe(1);

  const nc = new NearestCentroid().fit(X, y);
  expect(nc.predict(X).length).toBe(X.length);
});

