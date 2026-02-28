import type { Matrix, Vector } from "../types";
import { inverseMatrix } from "../utils/linalg";

export function featureMeans(X: Matrix): Vector {
  const nSamples = X.length;
  const nFeatures = X[0].length;
  const means = new Array<number>(nFeatures).fill(0);
  for (let i = 0; i < nSamples; i += 1) {
    for (let j = 0; j < nFeatures; j += 1) {
      means[j] += X[i][j];
    }
  }
  for (let j = 0; j < nFeatures; j += 1) {
    means[j] /= nSamples;
  }
  return means;
}

export function covarianceMatrix(X: Matrix, means: Vector): Matrix {
  const nSamples = X.length;
  const nFeatures = X[0].length;
  const cov: Matrix = Array.from({ length: nFeatures }, () =>
    new Array<number>(nFeatures).fill(0),
  );
  for (let i = 0; i < nFeatures; i += 1) {
    for (let j = i; j < nFeatures; j += 1) {
      let sum = 0;
      for (let row = 0; row < nSamples; row += 1) {
        sum += (X[row][i] - means[i]) * (X[row][j] - means[j]);
      }
      const value = sum / Math.max(1, nSamples - 1);
      cov[i][j] = value;
      cov[j][i] = value;
    }
  }
  return cov;
}

export function addDiagonal(matrix: Matrix, value: number): Matrix {
  const out = matrix.map((row) => row.slice());
  for (let i = 0; i < out.length; i += 1) {
    out[i][i] += value;
  }
  return out;
}

export function matrixDeterminant(matrix: Matrix): number {
  const n = matrix.length;
  const A = matrix.map((row) => row.slice());
  let det = 1;
  for (let i = 0; i < n; i += 1) {
    let pivot = i;
    let maxAbs = Math.abs(A[i][i]);
    for (let r = i + 1; r < n; r += 1) {
      const candidate = Math.abs(A[r][i]);
      if (candidate > maxAbs) {
        maxAbs = candidate;
        pivot = r;
      }
    }
    if (maxAbs <= 1e-12) {
      return 0;
    }
    if (pivot !== i) {
      const tmp = A[i];
      A[i] = A[pivot];
      A[pivot] = tmp;
      det *= -1;
    }
    const pivotValue = A[i][i];
    det *= pivotValue;
    for (let r = i + 1; r < n; r += 1) {
      const factor = A[r][i] / pivotValue;
      for (let c = i; c < n; c += 1) {
        A[r][c] -= factor * A[i][c];
      }
    }
  }
  return det;
}

export function regularizedPrecision(covariance: Matrix): Matrix {
  let jitter = 1e-9;
  for (let attempt = 0; attempt < 8; attempt += 1) {
    try {
      return inverseMatrix(addDiagonal(covariance, jitter));
    } catch {
      jitter *= 10;
    }
  }
  throw new Error("Unable to invert covariance matrix; covariance may be singular.");
}

export function mahalanobisDistanceSquared(
  row: Vector,
  location: Vector,
  precision: Matrix,
): number {
  const diff = new Array<number>(row.length);
  for (let i = 0; i < row.length; i += 1) {
    diff[i] = row[i] - location[i];
  }
  const tmp = new Array<number>(row.length).fill(0);
  for (let i = 0; i < precision.length; i += 1) {
    for (let j = 0; j < precision[i].length; j += 1) {
      tmp[i] += precision[i][j] * diff[j];
    }
  }
  let sum = 0;
  for (let i = 0; i < diff.length; i += 1) {
    sum += diff[i] * tmp[i];
  }
  return sum;
}
