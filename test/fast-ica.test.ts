import { expect, test } from "bun:test";
import { FastICA } from "../src";

function meanSquaredError(a: number[][], b: number[][]): number {
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

test("FastICA transform/inverseTransform round-trips mixed signals", () => {
  const X: number[][] = [];
  for (let i = 0; i < 400; i += 1) {
    const t = (i / 400) * 8 * Math.PI;
    const s1 = Math.sin(t);
    const s2 = Math.sign(Math.sin(2.3 * t));
    const x1 = 1.0 * s1 + 0.5 * s2;
    const x2 = 0.4 * s1 + 1.2 * s2;
    X.push([x1, x2]);
  }

  const ica = new FastICA({ nComponents: 2, maxIter: 500, tolerance: 1e-5, randomState: 42 });
  const S = ica.fitTransform(X);
  expect(S.length).toBe(X.length);
  expect(S[0].length).toBe(2);
  expect(ica.components_).not.toBeNull();
  expect(ica.mixing_).not.toBeNull();

  const reconstructed = ica.inverseTransform(S);
  expect(meanSquaredError(X, reconstructed)).toBeLessThan(0.05);
});

test("FastICA is deterministic for fixed randomState", () => {
  const X = [
    [1.0, 0.5],
    [0.9, 0.4],
    [0.2, 1.2],
    [0.1, 1.1],
    [1.1, 0.6],
    [0.0, 1.0],
  ];

  const a = new FastICA({ nComponents: 2, randomState: 17, maxIter: 300 }).fit(X);
  const b = new FastICA({ nComponents: 2, randomState: 17, maxIter: 300 }).fit(X);
  expect(a.components_).toEqual(b.components_);
  expect(a.transform(X)).toEqual(b.transform(X));
});

test("FastICA validates fit state and dimensions", () => {
  const ica = new FastICA({ nComponents: 2 });
  expect(() => ica.transform([[1, 2]])).toThrow(/has not been fitted/i);

  ica.fit([
    [1, 2],
    [2, 4],
    [3, 6],
  ]);
  expect(() => ica.transform([[1, 2, 3]])).toThrow(/feature size mismatch/i);
});
