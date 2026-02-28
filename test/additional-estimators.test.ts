import { expect, test } from "bun:test";
import {
  AffinityPropagation,
  FactorAnalysis,
  IncrementalPCA,
  MeanShift,
  MiniBatchKMeans,
  MiniBatchNMF,
} from "../src";

test("MiniBatchKMeans clusters simple two-group data", () => {
  const X = [
    [0, 0],
    [0.1, -0.1],
    [0.2, 0.1],
    [5, 5],
    [5.1, 4.9],
    [4.9, 5.2],
  ];
  const km = new MiniBatchKMeans({ nClusters: 2, batchSize: 3, maxIter: 120, randomState: 7 }).fit(X);
  expect(km.clusterCenters_).not.toBeNull();
  expect(km.labels_?.length).toBe(X.length);
});

test("MeanShift and AffinityPropagation produce cluster labels", () => {
  const X = [
    [0, 0],
    [0.2, 0.1],
    [-0.1, 0.2],
    [4.8, 5.1],
    [5.0, 4.9],
    [5.2, 5.0],
  ];
  const ms = new MeanShift({ bandwidth: 1.2, maxIter: 50 }).fit(X);
  expect(ms.clusterCenters_!.length).toBeGreaterThan(0);
  expect(ms.labels_!.length).toBe(X.length);

  const ap = new AffinityPropagation({ damping: 0.7, maxIter: 120 }).fit(X);
  expect(ap.clusterCentersIndices_!.length).toBeGreaterThan(0);
  expect(ap.labels_!.length).toBe(X.length);
});

test("IncrementalPCA supports partialFit and transform", () => {
  const X = [
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 6],
  ];
  const ipca = new IncrementalPCA({ nComponents: 2, batchSize: 2 });
  ipca.partialFit(X.slice(0, 2));
  ipca.partialFit(X.slice(2));
  const emb = ipca.transform(X);
  expect(emb.length).toBe(X.length);
  expect(emb[0].length).toBe(2);
});

test("FactorAnalysis transforms and inverse-transforms", () => {
  const X = [
    [1, 2, 0],
    [2, 3, 1],
    [3, 4, 1],
    [4, 5, 2],
    [5, 6, 2],
  ];
  const fa = new FactorAnalysis({ nComponents: 2 }).fit(X);
  const latent = fa.transform(X);
  expect(latent.length).toBe(X.length);
  expect(latent[0].length).toBe(2);
  const recon = fa.inverseTransform(latent);
  expect(recon.length).toBe(X.length);
  expect(recon[0].length).toBe(X[0].length);
});

test("MiniBatchNMF fits non-negative data", () => {
  const X = [
    [1, 0.5, 0.2],
    [0.8, 0.4, 0.1],
    [0.2, 0.9, 1.1],
    [0.1, 1.0, 1.2],
  ];
  const nmf = new MiniBatchNMF({
    nComponents: 2,
    batchSize: 2,
    maxIter: 120,
    randomState: 13,
  }).fit(X);
  const W = nmf.transform(X);
  expect(W.length).toBe(X.length);
  expect(W[0].length).toBe(2);
});
