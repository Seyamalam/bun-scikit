import { expect, test } from "bun:test";
import { TruncatedSVD } from "../src";

test("TruncatedSVD extracts low-rank components and reconstructs input", () => {
  const X = [
    [1, 2, 3],
    [2, 4, 6],
    [3, 6, 9],
    [4, 8, 12],
    [5, 10, 15],
  ];

  const svd = new TruncatedSVD({ nComponents: 1, nIter: 20, randomState: 42 });
  const transformed = svd.fitTransform(X);

  expect(transformed.length).toBe(X.length);
  expect(transformed[0].length).toBe(1);
  expect(svd.components_).not.toBeNull();
  expect(svd.singularValues_).not.toBeNull();
  expect(svd.explainedVarianceRatio_![0]).toBeGreaterThan(0.99);

  const reconstructed = svd.inverseTransform(transformed);
  for (let i = 0; i < X.length; i += 1) {
    expect(reconstructed[i][0] / X[i][0]).toBeCloseTo(1, 2);
    expect(reconstructed[i][1] / X[i][1]).toBeCloseTo(1, 2);
    expect(reconstructed[i][2] / X[i][2]).toBeCloseTo(1, 2);
  }
});

test("TruncatedSVD is deterministic for fixed randomState", () => {
  const X = [
    [1, 0, 3, 2],
    [2, 1, 6, 4],
    [3, 0, 9, 6],
    [4, 1, 12, 8],
    [5, 0, 15, 10],
  ];
  const a = new TruncatedSVD({ nComponents: 2, nIter: 25, randomState: 7 }).fit(X);
  const b = new TruncatedSVD({ nComponents: 2, nIter: 25, randomState: 7 }).fit(X);

  expect(a.components_).toEqual(b.components_);
  expect(a.singularValues_).toEqual(b.singularValues_);
});

test("TruncatedSVD validates component bounds", () => {
  const X = [
    [1, 2],
    [3, 4],
  ];

  expect(() => new TruncatedSVD({ nComponents: 0 })).toThrow(/ncomponents must be an integer >= 1/i);
  expect(() => new TruncatedSVD({ nComponents: 3 }).fit(X)).toThrow(/cannot exceed min\(nsamples, nfeatures\)/i);
});
