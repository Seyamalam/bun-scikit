import type { Matrix, Vector } from "../types";

export function squaredEuclideanDistance(a: Vector, b: Vector): number {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

export function rbfKernel(a: Vector, b: Vector, lengthScale: number): number {
  const dist2 = squaredEuclideanDistance(a, b);
  return Math.exp(-0.5 * dist2 / Math.max(1e-12, lengthScale * lengthScale));
}

export function kernelMatrix(X: Matrix, Y: Matrix, lengthScale: number): Matrix {
  const out: Matrix = new Array(X.length);
  for (let i = 0; i < X.length; i += 1) {
    const row = new Array<number>(Y.length);
    for (let j = 0; j < Y.length; j += 1) {
      row[j] = rbfKernel(X[i], Y[j], lengthScale);
    }
    out[i] = row;
  }
  return out;
}

