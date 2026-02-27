import type { Matrix, Vector } from "../types";

export function uniqueSortedLabels(y: Vector): Vector {
  return Array.from(new Set(y)).sort((a, b) => a - b);
}

export function buildLabelIndex(labels: Vector): Map<number, number> {
  const out = new Map<number, number>();
  for (let i = 0; i < labels.length; i += 1) {
    out.set(labels[i], i);
  }
  return out;
}

export function argmax(values: number[]): number {
  let bestIndex = 0;
  let bestValue = values[0];
  for (let i = 1; i < values.length; i += 1) {
    if (values[i] > bestValue) {
      bestValue = values[i];
      bestIndex = i;
    }
  }
  return bestIndex;
}

export function normalizeProbabilitiesInPlace(row: number[]): void {
  let sum = 0;
  for (let i = 0; i < row.length; i += 1) {
    const value = Number.isFinite(row[i]) && row[i] >= 0 ? row[i] : 0;
    row[i] = value;
    sum += value;
  }
  if (sum <= 0) {
    const uniform = row.length === 0 ? 0 : 1 / row.length;
    for (let i = 0; i < row.length; i += 1) {
      row[i] = uniform;
    }
    return;
  }
  for (let i = 0; i < row.length; i += 1) {
    row[i] /= sum;
  }
}

export function selectColumns(X: Matrix, indices: number[]): Matrix {
  const out: Matrix = new Array(X.length);
  for (let i = 0; i < X.length; i += 1) {
    const row = new Array<number>(indices.length);
    for (let j = 0; j < indices.length; j += 1) {
      row[j] = X[i][indices[j]];
    }
    out[i] = row;
  }
  return out;
}
