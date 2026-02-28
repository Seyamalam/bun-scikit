import { expect, test } from "bun:test";
import {
  DictionaryLearning,
  MiniBatchDictionaryLearning,
  MiniBatchSparsePCA,
  SparsePCA,
} from "../src";

const X = [
  [1.0, 0.1, 0.2],
  [0.9, 0.2, 0.1],
  [0.2, 1.0, 0.9],
  [0.1, 0.9, 1.1],
  [0.5, 0.6, 0.4],
];

test("SparsePCA and MiniBatchSparsePCA expose decomposition transforms", () => {
  const spca = new SparsePCA({ nComponents: 2, alpha: 0.2 }).fit(X);
  const emb = spca.transform(X);
  expect(emb.length).toBe(X.length);
  expect(emb[0].length).toBe(2);
  const inv = spca.inverseTransform(emb);
  expect(inv.length).toBe(X.length);

  const mb = new MiniBatchSparsePCA({ nComponents: 2, alpha: 0.2, batchSize: 3 }).fit(X);
  const embMb = mb.transform(X);
  expect(embMb.length).toBe(X.length);
  expect(embMb[0].length).toBe(2);
});

test("DictionaryLearning and MiniBatchDictionaryLearning fit/transform/inverseTransform", () => {
  const dl = new DictionaryLearning({ nComponents: 2, alpha: 0.2 }).fit(X);
  const code = dl.transform(X);
  expect(code.length).toBe(X.length);
  expect(code[0].length).toBe(2);
  const recon = dl.inverseTransform(code);
  expect(recon.length).toBe(X.length);
  expect(recon[0].length).toBe(X[0].length);

  const mb = new MiniBatchDictionaryLearning({ nComponents: 2, alpha: 0.2, batchSize: 3 }).fit(X);
  const codeMb = mb.transform(X);
  expect(codeMb.length).toBe(X.length);
  expect(codeMb[0].length).toBe(2);
});
