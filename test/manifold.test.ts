import { expect, test } from "bun:test";
import {
  Isomap,
  LocallyLinearEmbedding,
  MDS,
  TSNE,
} from "../src";

const X = [
  [0, 0, 1],
  [0.2, -0.1, 0.9],
  [1.1, 1.0, 0.1],
  [1.2, 1.1, 0.2],
  [2.0, 2.1, -0.2],
  [2.1, 2.0, -0.1],
];

test("TSNE fitTransform returns requested dimensionality", () => {
  const emb = new TSNE({ nComponents: 2, randomState: 42 }).fitTransform(X);
  expect(emb.length).toBe(X.length);
  expect(emb[0].length).toBe(2);
});

test("MDS fits euclidean data", () => {
  const emb = new MDS({ nComponents: 2, randomState: 11 }).fitTransform(X);
  expect(emb.length).toBe(X.length);
  expect(emb[0].length).toBe(2);
});

test("Isomap and LLE produce embeddings and support transform", () => {
  const isomap = new Isomap({ nNeighbors: 2, nComponents: 2 }).fit(X);
  const isoEmb = isomap.transform([[0.1, 0.0, 0.95], [2.05, 2.0, -0.15]]);
  expect(isoEmb.length).toBe(2);
  expect(isoEmb[0].length).toBe(2);

  const lle = new LocallyLinearEmbedding({ nNeighbors: 2, nComponents: 2 }).fit(X);
  const lleEmb = lle.transform([[0.1, 0.0, 0.95], [2.05, 2.0, -0.15]]);
  expect(lleEmb.length).toBe(2);
  expect(lleEmb[0].length).toBe(2);
});
