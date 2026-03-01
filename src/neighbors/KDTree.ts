import type { Matrix, Vector } from "../types";
import { DistanceMetric, type DistanceMetricName } from "../metrics/DistanceMetric";
import { assertConsistentRowSize, assertFiniteMatrix, assertNonEmptyMatrix } from "../utils/validation";

export interface KDTreeOptions {
  leafSize?: number;
  metric?: DistanceMetricName;
  p?: number;
}

export interface KDTreeQueryResult {
  distances: Matrix;
  indices: number[][];
}

export interface KDTreeQueryRadiusResult {
  indices: number[][];
  distances?: Matrix;
}

export class KDTree {
  readonly data: Matrix;
  readonly leafSize: number;
  readonly metric: DistanceMetric;
  readonly nFeaturesIn_: number;

  constructor(X: Matrix, options: KDTreeOptions = {}) {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    this.data = X.map((row) => row.slice());
    this.nFeaturesIn_ = X[0].length;
    this.leafSize = options.leafSize ?? 40;
    if (!Number.isInteger(this.leafSize) || this.leafSize < 1) {
      throw new Error(`leafSize must be an integer >= 1. Got ${this.leafSize}.`);
    }
    this.metric = DistanceMetric.getMetric(options.metric ?? "minkowski", { p: options.p ?? 2 });
  }

  query(X: Matrix, k = 1): KDTreeQueryResult {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }
    if (!Number.isInteger(k) || k < 1) {
      throw new Error(`k must be an integer >= 1. Got ${k}.`);
    }

    const maxK = Math.min(k, this.data.length);
    const distances: Matrix = new Array(X.length);
    const indices: number[][] = new Array(X.length);

    for (let i = 0; i < X.length; i += 1) {
      const sorted = this.data
        .map((row, index) => ({ index, distance: this.metric.dist(X[i], row) }))
        .sort((a, b) => a.distance - b.distance)
        .slice(0, maxK);
      distances[i] = sorted.map((item) => item.distance);
      indices[i] = sorted.map((item) => item.index);
    }

    return { distances, indices };
  }

  queryRadius(
    X: Matrix,
    radius: number,
    countOnly = false,
    returnDistance = false,
    sortResults = false,
  ): number[] | KDTreeQueryRadiusResult {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }
    if (!Number.isFinite(radius) || radius <= 0) {
      throw new Error(`radius must be finite and > 0. Got ${radius}.`);
    }

    const counts = new Array<number>(X.length).fill(0);
    const allIndices: number[][] = new Array(X.length);
    const allDistances: Matrix = new Array(X.length);

    for (let i = 0; i < X.length; i += 1) {
      const matches: Array<{ index: number; distance: number }> = [];
      for (let j = 0; j < this.data.length; j += 1) {
        const distance = this.metric.dist(X[i], this.data[j]);
        if (distance <= radius) {
          matches.push({ index: j, distance });
        }
      }
      if (sortResults) {
        matches.sort((a, b) => a.distance - b.distance);
      }
      counts[i] = matches.length;
      allIndices[i] = matches.map((item) => item.index);
      allDistances[i] = matches.map((item) => item.distance);
    }

    if (countOnly) {
      return counts;
    }
    if (returnDistance) {
      return { indices: allIndices, distances: allDistances };
    }
    return { indices: allIndices };
  }

  kernelDensity(X: Matrix, bandwidth: number, kernel: "gaussian" | "tophat" = "gaussian"): Vector {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }
    if (!Number.isFinite(bandwidth) || bandwidth <= 0) {
      throw new Error(`bandwidth must be finite and > 0. Got ${bandwidth}.`);
    }

    const invBandwidth = 1 / bandwidth;
    const out = new Array<number>(X.length).fill(0);
    for (let i = 0; i < X.length; i += 1) {
      let density = 0;
      for (let j = 0; j < this.data.length; j += 1) {
        const scaledDistance = this.metric.dist(X[i], this.data[j]) * invBandwidth;
        if (kernel === "gaussian") {
          density += Math.exp(-0.5 * scaledDistance * scaledDistance);
        } else {
          density += scaledDistance <= 1 ? 1 : 0;
        }
      }
      out[i] = density / this.data.length;
    }
    return out;
  }

  getArrays(): { data: Matrix } {
    return { data: this.data.map((row) => row.slice()) };
  }
}

