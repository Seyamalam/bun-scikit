import { expect, test } from "bun:test";
import { NMF } from "../src";

function mse(a: number[][], b: number[][]): number {
  let total = 0;
  let count = 0;
  for (let i = 0; i < a.length; i += 1) {
    for (let j = 0; j < a[i].length; j += 1) {
      const diff = a[i][j] - b[i][j];
      total += diff * diff;
      count += 1;
    }
  }
  return total / count;
}

test("NMF factorizes and reconstructs a nonnegative matrix", () => {
  const X = [
    [1.0, 0.5, 0.2],
    [0.9, 0.4, 0.3],
    [0.2, 0.8, 1.0],
    [0.1, 0.9, 1.1],
  ];

  const nmf = new NMF({ nComponents: 2, maxIter: 500, tolerance: 1e-6, randomState: 42 });
  const W = nmf.fitTransform(X);
  expect(W.length).toBe(X.length);
  expect(W[0].length).toBe(2);
  expect(nmf.components_).not.toBeNull();
  expect(nmf.reconstructionErr_).not.toBeNull();

  const reconstructed = nmf.inverseTransform(W);
  expect(mse(X, reconstructed)).toBeLessThan(0.03);
});

test("NMF transform validates nonnegative inputs", () => {
  const nmf = new NMF({ nComponents: 2, randomState: 1 }).fit([
    [1, 2],
    [3, 4],
  ]);
  expect(() => nmf.transform([[1, -1]])).toThrow(/non-negative/i);
});
