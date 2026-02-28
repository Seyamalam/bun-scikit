import type { Matrix } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";
import {
  dot,
  inverseMatrix,
  multiplyMatrices,
  solveSymmetricPositiveDefinite,
  transpose,
} from "../utils/linalg";

export interface LocallyLinearEmbeddingOptions {
  nNeighbors?: number;
  nComponents?: number;
  reg?: number;
}

export class LocallyLinearEmbedding {
  embedding_: Matrix | null = null;
  reconstructionError_: number | null = null;
  nFeaturesIn_: number | null = null;

  private nNeighbors: number;
  private nComponents: number;
  private reg: number;
  private XTrain: Matrix | null = null;

  constructor(options: LocallyLinearEmbeddingOptions = {}) {
    this.nNeighbors = options.nNeighbors ?? 5;
    this.nComponents = options.nComponents ?? 2;
    this.reg = options.reg ?? 1e-3;
    if (!Number.isInteger(this.nNeighbors) || this.nNeighbors < 1) {
      throw new Error(`nNeighbors must be an integer >= 1. Got ${this.nNeighbors}.`);
    }
    if (!Number.isInteger(this.nComponents) || this.nComponents < 1) {
      throw new Error(`nComponents must be an integer >= 1. Got ${this.nComponents}.`);
    }
    if (!Number.isFinite(this.reg) || this.reg < 0) {
      throw new Error(`reg must be finite and >= 0. Got ${this.reg}.`);
    }
  }

  fit(X: Matrix): this {
    this.embedding_ = this.fitTransform(X);
    return this;
  }

  fitTransform(X: Matrix): Matrix {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (this.nNeighbors >= X.length) {
      throw new Error(`nNeighbors must be < nSamples (${X.length}). Got ${this.nNeighbors}.`);
    }
    if (this.nComponents >= X.length) {
      throw new Error(`nComponents must be < nSamples (${X.length}). Got ${this.nComponents}.`);
    }

    this.nFeaturesIn_ = X[0].length;
    this.XTrain = X.map((row) => row.slice());

    const neighbors = nearestNeighborIndices(X, this.nNeighbors);
    const weights = reconstructionWeights(X, neighbors, this.reg);
    const M = lleCostMatrix(X.length, neighbors, weights);
    const eigen = jacobiEigenDecomposition(M);
    const order = Array.from({ length: eigen.eigenvalues.length }, (_, idx) => idx).sort(
      (a, b) => eigen.eigenvalues[a] - eigen.eigenvalues[b],
    );

    const embedding: Matrix = Array.from({ length: X.length }, () =>
      new Array<number>(this.nComponents).fill(0),
    );
    const scale = Math.sqrt(X.length);
    for (let c = 0; c < this.nComponents; c += 1) {
      const eigenIndex = order[c + 1];
      for (let i = 0; i < X.length; i += 1) {
        embedding[i][c] = eigen.eigenvectors[i][eigenIndex] * scale;
      }
    }

    this.embedding_ = embedding.map((row) => row.slice());
    this.reconstructionError_ = computeReconstructionError(X, neighbors, weights);
    return embedding;
  }

  transform(X: Matrix): Matrix {
    if (!this.embedding_ || !this.XTrain || this.nFeaturesIn_ === null) {
      throw new Error("LocallyLinearEmbedding has not been fitted.");
    }
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }

    const out: Matrix = new Array(X.length);
    for (let i = 0; i < X.length; i += 1) {
      const neighbors = nearestNeighborIndicesForPoint(X[i], this.XTrain, this.nNeighbors);
      const weights = barycentricWeights(X[i], this.XTrain, neighbors, this.reg);
      const row = new Array<number>(this.embedding_[0].length).fill(0);
      for (let k = 0; k < neighbors.length; k += 1) {
        const emb = this.embedding_[neighbors[k]];
        const weight = weights[k];
        for (let c = 0; c < row.length; c += 1) {
          row[c] += weight * emb[c];
        }
      }
      out[i] = row;
    }

    return out;
  }
}

function squaredEuclideanDistance(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

function nearestNeighborIndices(X: Matrix, nNeighbors: number): Matrix {
  const out: Matrix = new Array(X.length);
  for (let i = 0; i < X.length; i += 1) {
    const sorted = Array.from({ length: X.length }, (_, idx) => idx)
      .filter((idx) => idx !== i)
      .sort((a, b) => squaredEuclideanDistance(X[i], X[a]) - squaredEuclideanDistance(X[i], X[b]))
      .slice(0, nNeighbors);
    out[i] = sorted;
  }
  return out;
}

function nearestNeighborIndicesForPoint(point: number[], XTrain: Matrix, nNeighbors: number): number[] {
  return Array.from({ length: XTrain.length }, (_, idx) => idx)
    .sort((a, b) => squaredEuclideanDistance(point, XTrain[a]) - squaredEuclideanDistance(point, XTrain[b]))
    .slice(0, nNeighbors);
}

function barycentricWeights(
  point: number[],
  XTrain: Matrix,
  neighborIndices: number[],
  reg: number,
): number[] {
  const k = neighborIndices.length;
  const diffs: Matrix = new Array(k);
  for (let i = 0; i < k; i += 1) {
    const diff = new Array<number>(point.length);
    const neighbor = XTrain[neighborIndices[i]];
    for (let j = 0; j < point.length; j += 1) {
      diff[j] = point[j] - neighbor[j];
    }
    diffs[i] = diff;
  }

  const covariance: Matrix = Array.from({ length: k }, () => new Array<number>(k).fill(0));
  let trace = 0;
  for (let i = 0; i < k; i += 1) {
    for (let j = i; j < k; j += 1) {
      const value = dot(diffs[i], diffs[j]);
      covariance[i][j] = value;
      covariance[j][i] = value;
    }
    trace += covariance[i][i];
  }
  const regularization = reg * (trace > 1e-12 ? trace : 1);
  for (let i = 0; i < k; i += 1) {
    covariance[i][i] += regularization;
  }

  const ones = new Array<number>(k).fill(1);
  let weights: number[];
  try {
    weights = solveSymmetricPositiveDefinite(covariance, ones);
  } catch {
    const inverse = inverseMatrix(covariance);
    weights = new Array<number>(k).fill(0);
    for (let i = 0; i < k; i += 1) {
      let sum = 0;
      for (let j = 0; j < k; j += 1) {
        sum += inverse[i][j] * ones[j];
      }
      weights[i] = sum;
    }
  }

  let normalizer = 0;
  for (let i = 0; i < k; i += 1) {
    normalizer += weights[i];
  }
  if (Math.abs(normalizer) <= 1e-12) {
    return new Array<number>(k).fill(1 / k);
  }
  for (let i = 0; i < k; i += 1) {
    weights[i] /= normalizer;
  }
  return weights;
}

function reconstructionWeights(X: Matrix, neighbors: Matrix, reg: number): Matrix {
  const out: Matrix = new Array(X.length);
  for (let i = 0; i < X.length; i += 1) {
    out[i] = barycentricWeights(X[i], X, neighbors[i] as number[], reg);
  }
  return out;
}

function lleCostMatrix(nSamples: number, neighbors: Matrix, weights: Matrix): Matrix {
  const W: Matrix = Array.from({ length: nSamples }, () => new Array<number>(nSamples).fill(0));
  for (let i = 0; i < nSamples; i += 1) {
    const rowNeighbors = neighbors[i] as number[];
    const rowWeights = weights[i];
    for (let k = 0; k < rowNeighbors.length; k += 1) {
      W[i][rowNeighbors[k]] = rowWeights[k];
    }
  }

  const B: Matrix = Array.from({ length: nSamples }, (_, i) => {
    const row = new Array<number>(nSamples);
    for (let j = 0; j < nSamples; j += 1) {
      row[j] = (i === j ? 1 : 0) - W[i][j];
    }
    return row;
  });

  return multiplyMatrices(transpose(B), B);
}

function computeReconstructionError(X: Matrix, neighbors: Matrix, weights: Matrix): number {
  let sum = 0;
  for (let i = 0; i < X.length; i += 1) {
    const recon = new Array<number>(X[0].length).fill(0);
    const rowNeighbors = neighbors[i] as number[];
    for (let k = 0; k < rowNeighbors.length; k += 1) {
      const neighbor = X[rowNeighbors[k]];
      const weight = weights[i][k];
      for (let j = 0; j < recon.length; j += 1) {
        recon[j] += weight * neighbor[j];
      }
    }
    for (let j = 0; j < recon.length; j += 1) {
      const diff = X[i][j] - recon[j];
      sum += diff * diff;
    }
  }
  return sum / X.length;
}

function jacobiEigenDecomposition(matrix: Matrix): { eigenvalues: number[]; eigenvectors: Matrix } {
  const n = matrix.length;
  const A = matrix.map((row) => row.slice());
  const V: Matrix = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
  );

  for (let iter = 0; iter < 10_000; iter += 1) {
    let p = 0;
    let q = 1;
    let maxValue = 0;
    for (let i = 0; i < n; i += 1) {
      for (let j = i + 1; j < n; j += 1) {
        const value = Math.abs(A[i][j]);
        if (value > maxValue) {
          maxValue = value;
          p = i;
          q = j;
        }
      }
    }
    if (maxValue <= 1e-12) {
      break;
    }

    const app = A[p][p];
    const aqq = A[q][q];
    const apq = A[p][q];
    const phi = 0.5 * Math.atan2(2 * apq, aqq - app);
    const c = Math.cos(phi);
    const s = Math.sin(phi);

    for (let i = 0; i < n; i += 1) {
      if (i === p || i === q) {
        continue;
      }
      const aip = A[i][p];
      const aiq = A[i][q];
      A[i][p] = c * aip - s * aiq;
      A[p][i] = A[i][p];
      A[i][q] = s * aip + c * aiq;
      A[q][i] = A[i][q];
    }

    A[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
    A[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
    A[p][q] = 0;
    A[q][p] = 0;

    for (let i = 0; i < n; i += 1) {
      const vip = V[i][p];
      const viq = V[i][q];
      V[i][p] = c * vip - s * viq;
      V[i][q] = s * vip + c * viq;
    }
  }

  const eigenvalues = new Array<number>(n);
  for (let i = 0; i < n; i += 1) {
    eigenvalues[i] = A[i][i];
  }

  return { eigenvalues, eigenvectors: V };
}
