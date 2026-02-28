import type { Matrix, Vector } from "../types";
import { inverseMatrix } from "../utils/linalg";

export function classIndices(y: Vector): Map<number, number[]> {
  const out = new Map<number, number[]>();
  for (let i = 0; i < y.length; i += 1) {
    const label = y[i];
    const arr = out.get(label);
    if (arr) {
      arr.push(i);
    } else {
      out.set(label, [i]);
    }
  }
  return out;
}

export function featureMeans(X: Matrix): Vector {
  const means = new Array<number>(X[0].length).fill(0);
  for (let i = 0; i < X.length; i += 1) {
    for (let j = 0; j < X[i].length; j += 1) {
      means[j] += X[i][j];
    }
  }
  for (let j = 0; j < means.length; j += 1) {
    means[j] /= X.length;
  }
  return means;
}

export function covariance(X: Matrix, mean: Vector): Matrix {
  const nFeatures = X[0].length;
  const cov: Matrix = Array.from({ length: nFeatures }, () =>
    new Array<number>(nFeatures).fill(0),
  );

  for (let i = 0; i < X.length; i += 1) {
    for (let a = 0; a < nFeatures; a += 1) {
      const da = X[i][a] - mean[a];
      for (let b = a; b < nFeatures; b += 1) {
        cov[a][b] += da * (X[i][b] - mean[b]);
      }
    }
  }

  const denom = Math.max(1, X.length - 1);
  for (let a = 0; a < nFeatures; a += 1) {
    for (let b = a; b < nFeatures; b += 1) {
      cov[a][b] /= denom;
      cov[b][a] = cov[a][b];
    }
  }
  return cov;
}

export function regularizedInverse(cov: Matrix): Matrix {
  let jitter = 1e-9;
  for (let attempt = 0; attempt < 8; attempt += 1) {
    const matrix = cov.map((row) => row.slice());
    for (let i = 0; i < matrix.length; i += 1) {
      matrix[i][i] += jitter;
    }
    try {
      return inverseMatrix(matrix);
    } catch {
      jitter *= 10;
    }
  }
  throw new Error("Failed to invert covariance matrix.");
}

export function logDeterminant(cov: Matrix): number {
  const n = cov.length;
  const A = cov.map((row) => row.slice());
  let logDet = 0;

  for (let i = 0; i < n; i += 1) {
    let pivot = i;
    let maxAbs = Math.abs(A[i][i]);
    for (let r = i + 1; r < n; r += 1) {
      const value = Math.abs(A[r][i]);
      if (value > maxAbs) {
        maxAbs = value;
        pivot = r;
      }
    }
    if (maxAbs <= 1e-12) {
      return Number.NEGATIVE_INFINITY;
    }
    if (pivot !== i) {
      const tmp = A[i];
      A[i] = A[pivot];
      A[pivot] = tmp;
    }
    const diag = A[i][i];
    logDet += Math.log(Math.abs(diag));
    for (let r = i + 1; r < n; r += 1) {
      const factor = A[r][i] / diag;
      for (let c = i; c < n; c += 1) {
        A[r][c] -= factor * A[i][c];
      }
    }
  }

  return logDet;
}

export function quadraticForm(diff: Vector, precision: Matrix): number {
  const tmp = new Array<number>(diff.length).fill(0);
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
