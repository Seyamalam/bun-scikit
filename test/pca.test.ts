import { expect, test } from "bun:test";
import { PCA } from "../src";

function columnVariance(X: number[][], column: number): number {
  let mean = 0;
  for (let i = 0; i < X.length; i += 1) {
    mean += X[i][column];
  }
  mean /= X.length;

  let variance = 0;
  for (let i = 0; i < X.length; i += 1) {
    const centered = X[i][column] - mean;
    variance += centered * centered;
  }
  return variance / (X.length > 1 ? X.length - 1 : 1);
}

test("PCA captures a dominant 1D direction", () => {
  const X = [
    [1, 2],
    [2, 4],
    [3, 6],
    [4, 8],
    [5, 10],
  ];

  const pca = new PCA({ nComponents: 1 });
  const transformed = pca.fitTransform(X);

  expect(transformed.length).toBe(X.length);
  expect(transformed[0].length).toBe(1);
  expect(pca.components_).not.toBeNull();
  expect(pca.explainedVarianceRatio_).not.toBeNull();
  expect(pca.explainedVarianceRatio_![0]).toBeGreaterThan(0.999);

  const reconstructed = pca.inverseTransform(transformed);
  for (let i = 0; i < X.length; i += 1) {
    expect(reconstructed[i][0]).toBeCloseTo(X[i][0], 5);
    expect(reconstructed[i][1]).toBeCloseTo(X[i][1], 5);
  }
});

test("PCA whitening scales transformed components to unit variance", () => {
  const X = [
    [2.0, 0.5, 1.0],
    [2.2, 0.4, 0.8],
    [3.9, 0.9, 2.0],
    [4.1, 1.1, 2.2],
    [5.0, 1.5, 2.7],
    [5.1, 1.4, 2.9],
  ];

  const pca = new PCA({ nComponents: 2, whiten: true });
  const transformed = pca.fitTransform(X);

  expect(transformed[0].length).toBe(2);
  expect(columnVariance(transformed, 0)).toBeCloseTo(1, 6);
  expect(columnVariance(transformed, 1)).toBeCloseTo(1, 6);
});

test("PCA validates fit and transform constraints", () => {
  const pca = new PCA({ nComponents: 2 });

  expect(() => pca.transform([[1, 2]])).toThrow(/has not been fitted/i);
  pca.fit([
    [1, 2],
    [3, 4],
  ]);
  expect(() => pca.transform([[1, 2, 3]])).toThrow(/feature size mismatch/i);
});

test("PCA rejects invalid nComponents", () => {
  const X = [
    [1, 0],
    [0, 1],
  ];
  expect(() => new PCA({ nComponents: 0 })).toThrow(/must be an integer >= 1/i);
  expect(() => new PCA({ nComponents: 3 }).fit(X)).toThrow(/cannot exceed min\(nSamples, nFeatures\)/i);
});
