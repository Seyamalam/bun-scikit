import type { Matrix, Vector } from "../types";
import { inverseMatrix } from "../utils/linalg";

export function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussianRandom(random: () => number): number {
  const u1 = Math.max(1e-12, random());
  const u2 = random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

export function logSumExp(values: Vector): number {
  let maxValue = values[0];
  for (let i = 1; i < values.length; i += 1) {
    if (values[i] > maxValue) {
      maxValue = values[i];
    }
  }
  let sum = 0;
  for (let i = 0; i < values.length; i += 1) {
    sum += Math.exp(values[i] - maxValue);
  }
  return maxValue + Math.log(Math.max(1e-300, sum));
}

export function covarianceRegularized(cov: Matrix, regCovar: number): Matrix {
  const out = cov.map((row) => row.slice());
  for (let i = 0; i < out.length; i += 1) {
    out[i][i] += regCovar;
  }
  return out;
}

export function determinant(matrix: Matrix): number {
  const n = matrix.length;
  const A = matrix.map((row) => row.slice());
  let det = 1;
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
      return 0;
    }
    if (pivot !== i) {
      const tmp = A[i];
      A[i] = A[pivot];
      A[pivot] = tmp;
      det *= -1;
    }
    det *= A[i][i];
    for (let r = i + 1; r < n; r += 1) {
      const factor = A[r][i] / A[i][i];
      for (let c = i; c < n; c += 1) {
        A[r][c] -= factor * A[i][c];
      }
    }
  }
  return det;
}

export function inverseRegularized(covariance: Matrix): Matrix {
  let jitter = 1e-9;
  for (let attempt = 0; attempt < 8; attempt += 1) {
    try {
      const matrix = covariance.map((row) => row.slice());
      for (let i = 0; i < matrix.length; i += 1) {
        matrix[i][i] += jitter;
      }
      return inverseMatrix(matrix);
    } catch {
      jitter *= 10;
    }
  }
  throw new Error("Failed to invert covariance matrix.");
}

export function logGaussianPdf(x: Vector, mean: Vector, covariance: Matrix): number {
  const inv = inverseRegularized(covariance);
  const det = Math.max(1e-300, determinant(covariance));
  const diff = new Array<number>(x.length);
  for (let i = 0; i < x.length; i += 1) {
    diff[i] = x[i] - mean[i];
  }
  const tmp = new Array<number>(x.length).fill(0);
  for (let i = 0; i < inv.length; i += 1) {
    for (let j = 0; j < inv[i].length; j += 1) {
      tmp[i] += inv[i][j] * diff[j];
    }
  }
  let quad = 0;
  for (let i = 0; i < diff.length; i += 1) {
    quad += diff[i] * tmp[i];
  }
  return -0.5 * (x.length * Math.log(2 * Math.PI) + Math.log(det) + quad);
}
