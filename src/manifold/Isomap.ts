import type { Matrix, Vector } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";
import { MDS } from "./MDS";

export interface IsomapOptions {
  nNeighbors?: number;
  nComponents?: number;
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

function knnGraphDistances(dist: Matrix, nNeighbors: number): Matrix {
  const n = dist.length;
  const graph: Matrix = Array.from({ length: n }, () => new Array<number>(n).fill(Number.POSITIVE_INFINITY));
  for (let i = 0; i < n; i += 1) {
    graph[i][i] = 0;
    const order = Array.from({ length: n }, (_, idx) => idx)
      .filter((idx) => idx !== i)
      .sort((a, b) => dist[i][a] - dist[i][b])
      .slice(0, Math.min(nNeighbors, n - 1));
    for (let k = 0; k < order.length; k += 1) {
      const j = order[k];
      graph[i][j] = dist[i][j];
      graph[j][i] = dist[i][j];
    }
  }
  return graph;
}

function floydWarshall(graph: Matrix): Matrix {
  const n = graph.length;
  const out = graph.map((row) => row.slice());
  for (let k = 0; k < n; k += 1) {
    for (let i = 0; i < n; i += 1) {
      for (let j = 0; j < n; j += 1) {
        const via = out[i][k] + out[k][j];
        if (via < out[i][j]) {
          out[i][j] = via;
        }
      }
    }
  }
  return out;
}

export class Isomap {
  embedding_: Matrix | null = null;
  nFeaturesIn_: number | null = null;

  private nNeighbors: number;
  private nComponents: number;
  private XTrain: Matrix | null = null;

  constructor(options: IsomapOptions = {}) {
    this.nNeighbors = options.nNeighbors ?? 5;
    this.nComponents = options.nComponents ?? 2;
    if (!Number.isInteger(this.nNeighbors) || this.nNeighbors < 1) {
      throw new Error(`nNeighbors must be an integer >= 1. Got ${this.nNeighbors}.`);
    }
    if (!Number.isInteger(this.nComponents) || this.nComponents < 1) {
      throw new Error(`nComponents must be an integer >= 1. Got ${this.nComponents}.`);
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
    this.XTrain = X.map((row) => row.slice());
    this.nFeaturesIn_ = X[0].length;

    const dist = pairwiseDistanceMatrix(X);
    const graph = knnGraphDistances(dist, this.nNeighbors);
    const geodesic = floydWarshall(graph);
    // Replace disconnected paths with large finite penalties for numerical stability.
    for (let i = 0; i < geodesic.length; i += 1) {
      for (let j = 0; j < geodesic[i].length; j += 1) {
        if (!Number.isFinite(geodesic[i][j])) {
          geodesic[i][j] = 1e6;
        }
      }
    }

    const mds = new MDS({
      nComponents: this.nComponents,
      dissimilarity: "precomputed",
    });
    const embedding = mds.fitTransform(geodesic);
    this.embedding_ = embedding.map((row) => row.slice());
    return embedding;
  }

  transform(X: Matrix): Matrix {
    if (!this.embedding_ || !this.XTrain || this.nFeaturesIn_ === null) {
      throw new Error("Isomap has not been fitted.");
    }
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }

    // Lightweight out-of-sample extension: weighted average of nearest training embeddings.
    const k = Math.min(this.nNeighbors, this.XTrain.length);
    return X.map((row) => {
      const neighbors = this.XTrain!
        .map((trainRow, index) => ({ index, distance: euclideanDistance(row, trainRow) }))
        .sort((a, b) => a.distance - b.distance)
        .slice(0, k);
      const out = new Array<number>(this.embedding_![0].length).fill(0);
      let weightSum = 0;
      for (let i = 0; i < neighbors.length; i += 1) {
        const weight = 1 / Math.max(neighbors[i].distance, 1e-12);
        weightSum += weight;
        const emb = this.embedding_![neighbors[i].index];
        for (let c = 0; c < out.length; c += 1) {
          out[c] += weight * emb[c];
        }
      }
      if (weightSum > 0) {
        for (let c = 0; c < out.length; c += 1) {
          out[c] /= weightSum;
        }
      }
      return out;
    });
  }
}
