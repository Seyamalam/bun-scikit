import type { Matrix, Vector } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";

export interface NearestNeighborsOptions {
  nNeighbors?: number;
  radius?: number;
}

export interface NeighborQueryResult {
  distances: Matrix;
  indices: number[][];
}

function euclideanDistance(a: Vector, b: Vector): number {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

export class NearestNeighbors {
  nFeaturesIn_: number | null = null;

  private nNeighbors: number;
  private radius: number;
  private XTrain: Matrix | null = null;

  constructor(options: NearestNeighborsOptions = {}) {
    this.nNeighbors = options.nNeighbors ?? 5;
    this.radius = options.radius ?? 1;
    if (!Number.isInteger(this.nNeighbors) || this.nNeighbors < 1) {
      throw new Error(`nNeighbors must be an integer >= 1. Got ${this.nNeighbors}.`);
    }
    if (!Number.isFinite(this.radius) || this.radius <= 0) {
      throw new Error(`radius must be finite and > 0. Got ${this.radius}.`);
    }
  }

  fit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    this.XTrain = X.map((row) => row.slice());
    this.nFeaturesIn_ = X[0].length;
    return this;
  }

  kneighbors(X?: Matrix, nNeighbors?: number): NeighborQueryResult {
    this.assertFitted();
    const query = X ?? this.XTrain!;
    assertNonEmptyMatrix(query);
    assertConsistentRowSize(query);
    assertFiniteMatrix(query);
    if (query[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${query[0].length}.`);
    }
    const k = nNeighbors ?? this.nNeighbors;
    if (!Number.isInteger(k) || k < 1) {
      throw new Error(`nNeighbors must be an integer >= 1. Got ${k}.`);
    }
    const distances: Matrix = new Array(query.length);
    const indices: number[][] = new Array(query.length);
    for (let i = 0; i < query.length; i += 1) {
      const pairs = this.XTrain!
        .map((row, idx) => ({ idx, dist: euclideanDistance(query[i], row) }))
        .sort((a, b) => a.dist - b.dist)
        .slice(0, Math.min(k, this.XTrain!.length));
      distances[i] = pairs.map((pair) => pair.dist);
      indices[i] = pairs.map((pair) => pair.idx);
    }
    return { distances, indices };
  }

  radiusNeighbors(X?: Matrix, radius?: number): NeighborQueryResult {
    this.assertFitted();
    const query = X ?? this.XTrain!;
    assertNonEmptyMatrix(query);
    assertConsistentRowSize(query);
    assertFiniteMatrix(query);
    if (query[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${query[0].length}.`);
    }
    const r = radius ?? this.radius;
    if (!Number.isFinite(r) || r <= 0) {
      throw new Error(`radius must be finite and > 0. Got ${r}.`);
    }

    const distances: Matrix = new Array(query.length);
    const indices: number[][] = new Array(query.length);
    for (let i = 0; i < query.length; i += 1) {
      const pairs = this.XTrain!
        .map((row, idx) => ({ idx, dist: euclideanDistance(query[i], row) }))
        .filter((pair) => pair.dist <= r)
        .sort((a, b) => a.dist - b.dist);
      distances[i] = pairs.map((pair) => pair.dist);
      indices[i] = pairs.map((pair) => pair.idx);
    }
    return { distances, indices };
  }

  private assertFitted(): void {
    if (!this.XTrain || this.nFeaturesIn_ === null) {
      throw new Error("NearestNeighbors has not been fitted.");
    }
  }
}
