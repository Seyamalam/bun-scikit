import type { Matrix, Vector } from "../types";
import { assertConsistentRowSize, assertFiniteMatrix, assertNonEmptyMatrix } from "../utils/validation";

export type DistanceMetricName = "euclidean" | "manhattan" | "chebyshev" | "minkowski";

export interface DistanceMetricOptions {
  p?: number;
}

function euclideanDistance(a: Vector, b: Vector): number {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

function manhattanDistance(a: Vector, b: Vector): number {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    sum += Math.abs(a[i] - b[i]);
  }
  return sum;
}

function chebyshevDistance(a: Vector, b: Vector): number {
  let maxDistance = 0;
  for (let i = 0; i < a.length; i += 1) {
    const d = Math.abs(a[i] - b[i]);
    if (d > maxDistance) {
      maxDistance = d;
    }
  }
  return maxDistance;
}

function minkowskiDistance(a: Vector, b: Vector, p: number): number {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    sum += Math.pow(Math.abs(a[i] - b[i]), p);
  }
  return Math.pow(sum, 1 / p);
}

type DistanceFn = (a: Vector, b: Vector) => number;

export class DistanceMetric {
  metric: DistanceMetricName;
  p: number;
  private distanceFn: DistanceFn;

  constructor(metric: DistanceMetricName = "euclidean", options: DistanceMetricOptions = {}) {
    this.metric = metric;
    this.p = options.p ?? 2;
    this.distanceFn = euclideanDistance;
    this.configure();
  }

  static getMetric(metric: DistanceMetricName = "euclidean", options: DistanceMetricOptions = {}): DistanceMetric {
    return new DistanceMetric(metric, options);
  }

  getParams(): { metric: DistanceMetricName; p: number } {
    return { metric: this.metric, p: this.p };
  }

  setParams(params: Partial<{ metric: DistanceMetricName; p: number }>): this {
    if (params.metric !== undefined) {
      this.metric = params.metric;
    }
    if (params.p !== undefined) {
      this.p = params.p;
    }
    this.configure();
    return this;
  }

  pairwise(X: Matrix, Y?: Matrix): Matrix {
    assertNonEmptyMatrix(X, "X");
    assertConsistentRowSize(X, "X");
    assertFiniteMatrix(X, "X");
    const YInput = Y ?? X;
    assertNonEmptyMatrix(YInput, "Y");
    assertConsistentRowSize(YInput, "Y");
    assertFiniteMatrix(YInput, "Y");
    if (X[0].length !== YInput[0].length) {
      throw new Error(`Feature size mismatch. Expected ${X[0].length}, got ${YInput[0].length}.`);
    }

    const out: Matrix = new Array(X.length);
    for (let i = 0; i < X.length; i += 1) {
      const row = new Array<number>(YInput.length);
      for (let j = 0; j < YInput.length; j += 1) {
        row[j] = this.distanceFn(X[i], YInput[j]);
      }
      out[i] = row;
    }
    return out;
  }

  dist(a: Vector, b: Vector): number {
    if (a.length !== b.length) {
      throw new Error(`Vector size mismatch. Expected ${a.length}, got ${b.length}.`);
    }
    return this.distanceFn(a, b);
  }

  private configure(): void {
    if (!Number.isFinite(this.p) || this.p <= 0) {
      throw new Error(`p must be finite and > 0. Got ${this.p}.`);
    }
    if (this.metric === "euclidean") {
      this.distanceFn = euclideanDistance;
      return;
    }
    if (this.metric === "manhattan") {
      this.distanceFn = manhattanDistance;
      return;
    }
    if (this.metric === "chebyshev") {
      this.distanceFn = chebyshevDistance;
      return;
    }
    if (this.metric === "minkowski") {
      const p = this.p;
      this.distanceFn = (a, b) => minkowskiDistance(a, b, p);
      return;
    }
    throw new Error(`Unsupported metric '${this.metric}'.`);
  }
}
