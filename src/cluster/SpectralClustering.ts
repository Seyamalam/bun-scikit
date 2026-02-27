import type { Matrix, Vector } from "../types";
import { dot, multiplyMatrixVector } from "../utils/linalg";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";
import { KMeans } from "./KMeans";

export type SpectralAffinity = "rbf" | "nearest_neighbors" | "precomputed";

export interface SpectralClusteringOptions {
  nClusters?: number;
  affinity?: SpectralAffinity;
  gamma?: number;
  nNeighbors?: number;
  nInit?: number;
  maxIter?: number;
  randomState?: number;
}

class Mulberry32 {
  private state: number;

  constructor(seed: number) {
    this.state = seed >>> 0;
  }

  next(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let t = this.state ^ (this.state >>> 15);
    t = Math.imul(t, this.state | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
}

function squaredEuclideanDistance(a: Vector, b: Vector): number {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

function normalizeVector(v: Vector): Vector {
  const norm = Math.sqrt(dot(v, v));
  if (norm <= 1e-15) {
    return v.map(() => 0);
  }
  return v.map((value) => value / norm);
}

function orthogonalize(v: Vector, basis: Matrix): Vector {
  const out = v.slice();
  for (let i = 0; i < basis.length; i += 1) {
    const component = dot(out, basis[i]);
    for (let j = 0; j < out.length; j += 1) {
      out[j] -= component * basis[i][j];
    }
  }
  return out;
}

function topKEigenvectorsSymmetric(
  matrix: Matrix,
  k: number,
  maxIter: number,
  randomState: number | undefined,
): Matrix {
  const n = matrix.length;
  const vectors: Matrix = [];
  const seedBase = randomState ?? 13_579;

  for (let component = 0; component < k; component += 1) {
    const rng = new Mulberry32(seedBase + component * 104_729);
    let v = new Array<number>(n);
    for (let i = 0; i < n; i += 1) {
      v[i] = rng.next() * 2 - 1;
    }
    v = normalizeVector(orthogonalize(v, vectors));

    for (let iter = 0; iter < maxIter; iter += 1) {
      const mv = multiplyMatrixVector(matrix, v);
      const next = normalizeVector(orthogonalize(mv, vectors));
      let delta = 0;
      for (let i = 0; i < n; i += 1) {
        const diff = next[i] - v[i];
        delta += diff * diff;
      }
      v = next;
      if (delta < 1e-12) {
        break;
      }
    }
    vectors.push(v);
  }

  return vectors;
}

function rowNormalize(X: Matrix): Matrix {
  return X.map((row) => {
    let norm = 0;
    for (let i = 0; i < row.length; i += 1) {
      norm += row[i] * row[i];
    }
    norm = Math.sqrt(norm);
    if (norm <= 1e-15) {
      return row.map(() => 0);
    }
    return row.map((value) => value / norm);
  });
}

export class SpectralClustering {
  labels_: Vector | null = null;
  affinityMatrix_: Matrix | null = null;
  embedding_: Matrix | null = null;
  nFeaturesIn_: number | null = null;

  private nClusters: number;
  private affinity: SpectralAffinity;
  private gamma: number;
  private nNeighbors: number;
  private nInit: number;
  private maxIter: number;
  private randomState?: number;

  constructor(options: SpectralClusteringOptions = {}) {
    this.nClusters = options.nClusters ?? 8;
    this.affinity = options.affinity ?? "rbf";
    this.gamma = options.gamma ?? 1;
    this.nNeighbors = options.nNeighbors ?? 10;
    this.nInit = options.nInit ?? 10;
    this.maxIter = options.maxIter ?? 200;
    this.randomState = options.randomState;

    if (!Number.isInteger(this.nClusters) || this.nClusters < 1) {
      throw new Error(`nClusters must be an integer >= 1. Got ${this.nClusters}.`);
    }
    if (!Number.isFinite(this.gamma) || this.gamma <= 0) {
      throw new Error(`gamma must be finite and > 0. Got ${this.gamma}.`);
    }
    if (!Number.isInteger(this.nNeighbors) || this.nNeighbors < 1) {
      throw new Error(`nNeighbors must be an integer >= 1. Got ${this.nNeighbors}.`);
    }
    if (!Number.isInteger(this.nInit) || this.nInit < 1) {
      throw new Error(`nInit must be an integer >= 1. Got ${this.nInit}.`);
    }
  }

  fit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    const nSamples = X.length;
    if (this.nClusters > nSamples) {
      throw new Error(
        `nClusters (${this.nClusters}) cannot exceed sample count (${nSamples}).`,
      );
    }

    const affinityMatrix = this.buildAffinity(X);
    const normalizedSimilarity = this.buildNormalizedSimilarity(affinityMatrix);
    const eigenvectors = topKEigenvectorsSymmetric(
      normalizedSimilarity,
      this.nClusters,
      this.maxIter,
      this.randomState,
    );

    const embedding: Matrix = Array.from({ length: nSamples }, (_, rowIndex) => {
      const row = new Array<number>(this.nClusters).fill(0);
      for (let component = 0; component < this.nClusters; component += 1) {
        row[component] = eigenvectors[component][rowIndex];
      }
      return row;
    });

    const normalizedEmbedding = rowNormalize(embedding);
    const kmeans = new KMeans({
      nClusters: this.nClusters,
      nInit: this.nInit,
      maxIter: this.maxIter,
      randomState: this.randomState,
    }).fit(normalizedEmbedding);

    this.labels_ = kmeans.labels_!.slice();
    this.affinityMatrix_ = affinityMatrix;
    this.embedding_ = normalizedEmbedding;
    this.nFeaturesIn_ = this.affinity === "precomputed" ? X.length : X[0].length;
    return this;
  }

  fitPredict(X: Matrix): Vector {
    this.fit(X);
    return this.labels_!.slice();
  }

  private buildAffinity(X: Matrix): Matrix {
    if (this.affinity === "precomputed") {
      if (X.length !== X[0].length) {
        throw new Error("precomputed affinity matrix must be square.");
      }
      const affinity = X.map((row) => row.slice());
      for (let i = 0; i < affinity.length; i += 1) {
        affinity[i][i] = 1;
      }
      return affinity;
    }

    const n = X.length;
    const affinity: Matrix = Array.from({ length: n }, () => new Array<number>(n).fill(0));
    for (let i = 0; i < n; i += 1) {
      affinity[i][i] = 1;
    }

    if (this.affinity === "rbf") {
      for (let i = 0; i < n; i += 1) {
        for (let j = i + 1; j < n; j += 1) {
          const value = Math.exp(-this.gamma * squaredEuclideanDistance(X[i], X[j]));
          affinity[i][j] = value;
          affinity[j][i] = value;
        }
      }
      return affinity;
    }

    const k = Math.min(this.nNeighbors, Math.max(1, n - 1));
    for (let i = 0; i < n; i += 1) {
      const distances = new Array<{ index: number; distance: number }>(n - 1);
      let cursor = 0;
      for (let j = 0; j < n; j += 1) {
        if (i === j) {
          continue;
        }
        distances[cursor] = {
          index: j,
          distance: squaredEuclideanDistance(X[i], X[j]),
        };
        cursor += 1;
      }
      distances.sort((a, b) => a.distance - b.distance);
      for (let t = 0; t < k; t += 1) {
        const neighbor = distances[t].index;
        affinity[i][neighbor] = 1;
        affinity[neighbor][i] = 1;
      }
    }
    return affinity;
  }

  private buildNormalizedSimilarity(A: Matrix): Matrix {
    const n = A.length;
    const degree = new Array<number>(n).fill(0);
    for (let i = 0; i < n; i += 1) {
      let sum = 0;
      for (let j = 0; j < n; j += 1) {
        sum += A[i][j];
      }
      degree[i] = sum;
    }

    const DInvSqrt = degree.map((value) => (value <= 1e-15 ? 0 : 1 / Math.sqrt(value)));
    const similarity: Matrix = Array.from({ length: n }, () => new Array<number>(n).fill(0));
    for (let i = 0; i < n; i += 1) {
      for (let j = 0; j < n; j += 1) {
        similarity[i][j] = DInvSqrt[i] * A[i][j] * DInvSqrt[j];
      }
    }
    return similarity;
  }
}
