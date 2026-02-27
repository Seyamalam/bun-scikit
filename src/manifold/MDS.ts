import type { Matrix, Vector } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";
import { dot, multiplyMatrixVector } from "../utils/linalg";

export type MDSDissimilarity = "euclidean" | "precomputed";

export interface MDSOptions {
  nComponents?: number;
  dissimilarity?: MDSDissimilarity;
  randomState?: number;
  maxIter?: number;
}

function euclideanDistance(a: Vector, b: Vector): number {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

function pairwiseDistanceMatrix(X: Matrix): Matrix {
  const out: Matrix = Array.from({ length: X.length }, () => new Array<number>(X.length).fill(0));
  for (let i = 0; i < X.length; i += 1) {
    for (let j = i + 1; j < X.length; j += 1) {
      const dist = euclideanDistance(X[i], X[j]);
      out[i][j] = dist;
      out[j][i] = dist;
    }
  }
  return out;
}

function normalize(v: Vector): Vector {
  const norm = Math.sqrt(dot(v, v));
  if (norm <= 1e-12) {
    return v.map(() => 0);
  }
  return v.map((value) => value / norm);
}

function orthogonalize(v: Vector, basis: Matrix): Vector {
  const out = v.slice();
  for (let i = 0; i < basis.length; i += 1) {
    const projection = dot(out, basis[i]);
    for (let j = 0; j < out.length; j += 1) {
      out[j] -= projection * basis[i][j];
    }
  }
  return out;
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

function topEigenpairsSymmetric(matrix: Matrix, nComponents: number, maxIter: number, seed: number): {
  values: Vector;
  vectors: Matrix;
} {
  const n = matrix.length;
  const vectors: Matrix = [];
  const values = new Array<number>(nComponents).fill(0);
  for (let c = 0; c < nComponents; c += 1) {
    const rng = new Mulberry32(seed + c * 104_729);
    let v = new Array<number>(n);
    for (let i = 0; i < n; i += 1) {
      v[i] = rng.next() * 2 - 1;
    }
    v = normalize(orthogonalize(v, vectors));
    for (let iter = 0; iter < maxIter; iter += 1) {
      const mv = multiplyMatrixVector(matrix, v);
      const next = normalize(orthogonalize(mv, vectors));
      let delta = 0;
      for (let i = 0; i < n; i += 1) {
        const d = next[i] - v[i];
        delta += d * d;
      }
      v = next;
      if (delta < 1e-12) {
        break;
      }
    }
    const mv = multiplyMatrixVector(matrix, v);
    values[c] = dot(v, mv);
    vectors.push(v);
  }
  return { values, vectors };
}

function classicalMds(distanceMatrix: Matrix, nComponents: number, maxIter: number, seed: number): Matrix {
  const n = distanceMatrix.length;
  const squared: Matrix = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      squared[i][j] = distanceMatrix[i][j] ** 2;
    }
  }

  const rowMeans = new Array<number>(n).fill(0);
  const colMeans = new Array<number>(n).fill(0);
  let grandMean = 0;
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      rowMeans[i] += squared[i][j];
      colMeans[j] += squared[i][j];
      grandMean += squared[i][j];
    }
  }
  for (let i = 0; i < n; i += 1) {
    rowMeans[i] /= n;
    colMeans[i] /= n;
  }
  grandMean /= n * n;

  const B: Matrix = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      B[i][j] = -0.5 * (squared[i][j] - rowMeans[i] - colMeans[j] + grandMean);
    }
  }

  const eig = topEigenpairsSymmetric(B, Math.min(nComponents, n), maxIter, seed);
  const embedding: Matrix = Array.from({ length: n }, () => new Array<number>(eig.values.length).fill(0));
  for (let c = 0; c < eig.values.length; c += 1) {
    const lambda = Math.max(0, eig.values[c]);
    const scale = Math.sqrt(lambda);
    for (let i = 0; i < n; i += 1) {
      embedding[i][c] = eig.vectors[c][i] * scale;
    }
  }
  return embedding;
}

export class MDS {
  embedding_: Matrix | null = null;
  stress_: number | null = null;
  nFeaturesIn_: number | null = null;

  private nComponents: number;
  private dissimilarity: MDSDissimilarity;
  private randomState?: number;
  private maxIter: number;

  constructor(options: MDSOptions = {}) {
    this.nComponents = options.nComponents ?? 2;
    this.dissimilarity = options.dissimilarity ?? "euclidean";
    this.randomState = options.randomState;
    this.maxIter = options.maxIter ?? 500;
    if (!Number.isInteger(this.nComponents) || this.nComponents < 1) {
      throw new Error(`nComponents must be an integer >= 1. Got ${this.nComponents}.`);
    }
    if (!Number.isInteger(this.maxIter) || this.maxIter < 1) {
      throw new Error(`maxIter must be an integer >= 1. Got ${this.maxIter}.`);
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
    if (this.dissimilarity === "precomputed" && X.length !== X[0].length) {
      throw new Error("precomputed dissimilarity requires a square matrix.");
    }

    const dist = this.dissimilarity === "euclidean" ? pairwiseDistanceMatrix(X) : X.map((row) => row.slice());
    const embedding = classicalMds(dist, this.nComponents, this.maxIter, this.randomState ?? 0);
    this.embedding_ = embedding.map((row) => row.slice());
    this.nFeaturesIn_ = this.dissimilarity === "euclidean" ? X[0].length : X.length;
    this.stress_ = 0;
    return embedding;
  }
}
