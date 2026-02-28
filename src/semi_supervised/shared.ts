import type { Matrix, Vector } from "../types";
import {
  argmax,
  normalizeProbabilitiesInPlace,
  uniqueSortedLabels,
} from "../utils/classification";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  assertVectorLength,
} from "../utils/validation";

export type SemiSupervisedKernel = "rbf" | "knn";

export interface SemiSupervisedConfig {
  kernel: SemiSupervisedKernel;
  gamma: number;
  nNeighbors: number;
}

export interface SemiSupervisedState {
  classes: Vector;
  classToIndex: Map<number, number>;
  Y: Matrix;
  labeledMask: boolean[];
}

export function validateSemiSupervisedInputs(X: Matrix, y: Vector): void {
  if (X.length === 0) {
    throw new Error("X must be a non-empty 2D array.");
  }
  assertConsistentRowSize(X);
  assertFiniteMatrix(X);
  assertVectorLength(y, X.length);
  assertFiniteVector(y);
}

export function initializeState(y: Vector): SemiSupervisedState {
  const classes = uniqueSortedLabels(y.filter((value) => value !== -1));
  if (classes.length === 0) {
    throw new Error("At least one labeled sample is required (labels other than -1).");
  }
  const classToIndex = new Map<number, number>();
  for (let i = 0; i < classes.length; i += 1) {
    classToIndex.set(classes[i], i);
  }
  const Y: Matrix = new Array(y.length);
  const labeledMask = new Array<boolean>(y.length).fill(false);
  for (let i = 0; i < y.length; i += 1) {
    const row = new Array<number>(classes.length).fill(0);
    if (y[i] !== -1) {
      const idx = classToIndex.get(y[i]);
      if (idx === undefined) {
        throw new Error(`Unknown class label ${y[i]}.`);
      }
      row[idx] = 1;
      labeledMask[i] = true;
    }
    Y[i] = row;
  }
  return { classes, classToIndex, Y, labeledMask };
}

function squaredEuclideanDistance(a: Vector, b: Vector): number {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

function buildKnnAffinity(X: Matrix, nNeighbors: number): Matrix {
  const nSamples = X.length;
  const affinity: Matrix = Array.from({ length: nSamples }, () =>
    new Array<number>(nSamples).fill(0),
  );

  for (let i = 0; i < nSamples; i += 1) {
    const distances: Array<{ index: number; distance: number }> = [];
    for (let j = 0; j < nSamples; j += 1) {
      if (i === j) {
        continue;
      }
      distances.push({
        index: j,
        distance: squaredEuclideanDistance(X[i], X[j]),
      });
    }
    distances.sort((a, b) => a.distance - b.distance);
    const limit = Math.min(nNeighbors, distances.length);
    for (let j = 0; j < limit; j += 1) {
      const neighbor = distances[j].index;
      affinity[i][neighbor] = 1;
      affinity[neighbor][i] = 1;
    }
    affinity[i][i] = 1;
  }

  return affinity;
}

function buildRbfAffinity(X: Matrix, gamma: number): Matrix {
  const nSamples = X.length;
  const affinity: Matrix = Array.from({ length: nSamples }, () =>
    new Array<number>(nSamples).fill(0),
  );
  for (let i = 0; i < nSamples; i += 1) {
    for (let j = i; j < nSamples; j += 1) {
      const value = Math.exp(-gamma * squaredEuclideanDistance(X[i], X[j]));
      affinity[i][j] = value;
      affinity[j][i] = value;
    }
  }
  return affinity;
}

export function buildTransitionMatrix(X: Matrix, config: SemiSupervisedConfig): Matrix {
  const affinity =
    config.kernel === "knn"
      ? buildKnnAffinity(X, config.nNeighbors)
      : buildRbfAffinity(X, config.gamma);

  const out: Matrix = new Array(affinity.length);
  for (let i = 0; i < affinity.length; i += 1) {
    const row = affinity[i].slice();
    let sum = 0;
    for (let j = 0; j < row.length; j += 1) {
      sum += row[j];
    }
    if (sum <= 0) {
      row.fill(0);
      row[i] = 1;
      sum = 1;
    }
    for (let j = 0; j < row.length; j += 1) {
      row[j] /= sum;
    }
    out[i] = row;
  }
  return out;
}

export function multiply(A: Matrix, B: Matrix): Matrix {
  const n = A.length;
  const k = A[0].length;
  const m = B[0].length;
  const out: Matrix = Array.from({ length: n }, () => new Array<number>(m).fill(0));
  for (let i = 0; i < n; i += 1) {
    for (let t = 0; t < k; t += 1) {
      const a = A[i][t];
      if (a === 0) {
        continue;
      }
      for (let j = 0; j < m; j += 1) {
        out[i][j] += a * B[t][j];
      }
    }
  }
  return out;
}

export function maxAbsDiff(A: Matrix, B: Matrix): number {
  let maxDiff = 0;
  for (let i = 0; i < A.length; i += 1) {
    for (let j = 0; j < A[i].length; j += 1) {
      const diff = Math.abs(A[i][j] - B[i][j]);
      if (diff > maxDiff) {
        maxDiff = diff;
      }
    }
  }
  return maxDiff;
}

export function normalizeRowsInPlace(X: Matrix): void {
  for (let i = 0; i < X.length; i += 1) {
    normalizeProbabilitiesInPlace(X[i]);
  }
}

export function transductionFromDistributions(
  distributions: Matrix,
  classes: Vector,
): Vector {
  return distributions.map((row) => classes[argmax(row)]);
}

export function queryWeights(
  XQuery: Matrix,
  XTrain: Matrix,
  config: SemiSupervisedConfig,
): Matrix {
  const nQuery = XQuery.length;
  const nTrain = XTrain.length;
  const weights: Matrix = Array.from({ length: nQuery }, () => new Array<number>(nTrain).fill(0));

  for (let i = 0; i < nQuery; i += 1) {
    if (config.kernel === "knn") {
      const distances: Array<{ index: number; distance: number }> = [];
      for (let j = 0; j < nTrain; j += 1) {
        distances.push({
          index: j,
          distance: squaredEuclideanDistance(XQuery[i], XTrain[j]),
        });
      }
      distances.sort((a, b) => a.distance - b.distance);
      const k = Math.min(config.nNeighbors, distances.length);
      for (let j = 0; j < k; j += 1) {
        weights[i][distances[j].index] = 1;
      }
    } else {
      for (let j = 0; j < nTrain; j += 1) {
        weights[i][j] = Math.exp(
          -config.gamma * squaredEuclideanDistance(XQuery[i], XTrain[j]),
        );
      }
    }
    normalizeProbabilitiesInPlace(weights[i]);
  }

  return weights;
}
