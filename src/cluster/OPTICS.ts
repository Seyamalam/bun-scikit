import type { Matrix, Vector } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";
import { DBSCAN } from "./DBSCAN";

export type OPTICSClusterMethod = "dbscan" | "xi";

export interface OPTICSOptions {
  minSamples?: number;
  maxEps?: number;
  eps?: number;
  clusterMethod?: OPTICSClusterMethod;
}

function squaredEuclideanDistance(a: Vector, b: Vector): number {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

function pairwiseDistanceMatrix(X: Matrix): Matrix {
  const out: Matrix = Array.from({ length: X.length }, () => new Array<number>(X.length).fill(0));
  for (let i = 0; i < X.length; i += 1) {
    for (let j = i + 1; j < X.length; j += 1) {
      const distance = Math.sqrt(squaredEuclideanDistance(X[i], X[j]));
      out[i][j] = distance;
      out[j][i] = distance;
    }
  }
  return out;
}

function chooseDbscanEps(
  coreDistances: Vector,
  configuredEps: number | undefined,
  maxEps: number,
): number {
  if (configuredEps !== undefined) {
    return configuredEps;
  }
  if (Number.isFinite(maxEps)) {
    return maxEps;
  }
  const finite = coreDistances.filter((value) => Number.isFinite(value)).sort((a, b) => a - b);
  if (finite.length === 0) {
    return 0.5;
  }
  return finite[Math.floor((finite.length - 1) * 0.75)];
}

export class OPTICS {
  labels_: Vector | null = null;
  ordering_: Vector | null = null;
  reachability_: Vector | null = null;
  coreDistances_: Vector | null = null;
  predecessor_: Vector | null = null;
  nFeaturesIn_: number | null = null;

  private minSamples: number;
  private maxEps: number;
  private eps?: number;
  private clusterMethod: OPTICSClusterMethod;

  constructor(options: OPTICSOptions = {}) {
    this.minSamples = options.minSamples ?? 5;
    this.maxEps = options.maxEps ?? Number.POSITIVE_INFINITY;
    this.eps = options.eps;
    this.clusterMethod = options.clusterMethod ?? "dbscan";

    if (!Number.isInteger(this.minSamples) || this.minSamples < 2) {
      throw new Error(`minSamples must be an integer >= 2. Got ${this.minSamples}.`);
    }
    if (!(Number.isFinite(this.maxEps) || this.maxEps === Number.POSITIVE_INFINITY) || this.maxEps <= 0) {
      throw new Error(`maxEps must be > 0 or Infinity. Got ${this.maxEps}.`);
    }
    if (this.eps !== undefined && (!Number.isFinite(this.eps) || this.eps <= 0)) {
      throw new Error(`eps must be finite and > 0 when provided. Got ${this.eps}.`);
    }
  }

  fit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (this.minSamples > X.length) {
      throw new Error(`minSamples (${this.minSamples}) cannot exceed sample count (${X.length}).`);
    }

    const distances = pairwiseDistanceMatrix(X);
    const n = X.length;
    const coreDistances = new Array<number>(n).fill(Number.POSITIVE_INFINITY);
    for (let i = 0; i < n; i += 1) {
      const sorted = distances[i]
        .map((distance, index) => ({ distance, index }))
        .filter((item) => item.index !== i)
        .sort((a, b) => a.distance - b.distance);
      coreDistances[i] = sorted[this.minSamples - 2].distance;
    }

    const processed = new Array<boolean>(n).fill(false);
    const reachability = new Array<number>(n).fill(Number.POSITIVE_INFINITY);
    const predecessor = new Array<number>(n).fill(-1);
    const ordering: number[] = [];

    while (ordering.length < n) {
      let current = -1;
      let bestReach = Number.POSITIVE_INFINITY;
      for (let i = 0; i < n; i += 1) {
        if (!processed[i] && reachability[i] < bestReach) {
          bestReach = reachability[i];
          current = i;
        }
      }
      if (current === -1) {
        for (let i = 0; i < n; i += 1) {
          if (!processed[i]) {
            current = i;
            break;
          }
        }
      }

      processed[current] = true;
      ordering.push(current);

      const currentCore = coreDistances[current];
      if (!(Number.isFinite(currentCore) && currentCore <= this.maxEps)) {
        continue;
      }

      for (let neighbor = 0; neighbor < n; neighbor += 1) {
        if (processed[neighbor] || neighbor === current) {
          continue;
        }
        const distance = distances[current][neighbor];
        if (distance > this.maxEps) {
          continue;
        }
        const candidate = Math.max(currentCore, distance);
        if (candidate < reachability[neighbor]) {
          reachability[neighbor] = candidate;
          predecessor[neighbor] = current;
        }
      }
    }

    let labels: Vector;
    if (this.clusterMethod === "dbscan") {
      const eps = chooseDbscanEps(coreDistances, this.eps, this.maxEps);
      labels = new DBSCAN({ eps, minSamples: this.minSamples }).fit(X).labels_!.slice();
    } else {
      // Lightweight xi fallback: defer to DBSCAN-style extraction using the same epsilon selection.
      const eps = chooseDbscanEps(coreDistances, this.eps, this.maxEps);
      labels = new DBSCAN({ eps, minSamples: this.minSamples }).fit(X).labels_!.slice();
    }

    this.labels_ = labels;
    this.ordering_ = ordering;
    this.reachability_ = reachability;
    this.coreDistances_ = coreDistances;
    this.predecessor_ = predecessor;
    this.nFeaturesIn_ = X[0].length;
    return this;
  }

  fitPredict(X: Matrix): Vector {
    this.fit(X);
    return this.labels_!.slice();
  }
}
