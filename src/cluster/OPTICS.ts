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
  xi?: number;
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

function chooseXiEps(reachability: Vector, coreDistances: Vector, maxEps: number): number {
  if (Number.isFinite(maxEps)) {
    return maxEps;
  }
  const combined = new Array<number>();
  for (let i = 0; i < reachability.length; i += 1) {
    const value = Math.max(reachability[i], coreDistances[i]);
    if (Number.isFinite(value)) {
      combined.push(value);
    }
  }
  if (combined.length === 0) {
    return 1;
  }
  combined.sort((a, b) => a - b);
  return combined[Math.floor((combined.length - 1) * 0.9)];
}

function extractXiClusters(
  ordering: Vector,
  reachability: Vector,
  coreDistances: Vector,
  minSamples: number,
  xi: number,
  epsCutoff: number,
): Vector {
  const orderedReach = ordering.map((index) => reachability[index]);
  const orderedCore = ordering.map((index) => coreDistances[index]);
  const effective = orderedReach.map((value, idx) => Math.max(value, orderedCore[idx]));

  const labels = new Array<number>(ordering.length).fill(-1);
  let clusterId = 0;
  let segmentStart: number | null = null;

  for (let pos = 0; pos <= effective.length; pos += 1) {
    const inDenseRegion =
      pos < effective.length && Number.isFinite(effective[pos]) && effective[pos] <= epsCutoff;
    if (inDenseRegion) {
      if (segmentStart === null) {
        segmentStart = pos;
      }
      continue;
    }

    if (segmentStart !== null) {
      const segmentEnd = pos - 1;
      const segmentSize = segmentEnd - segmentStart + 1;
      if (segmentSize >= minSamples) {
        let valley = Number.POSITIVE_INFINITY;
        for (let i = segmentStart; i <= segmentEnd; i += 1) {
          if (effective[i] < valley) {
            valley = effective[i];
          }
        }

        const leftBoundary =
          segmentStart > 0 ? effective[segmentStart - 1] : Number.POSITIVE_INFINITY;
        const rightBoundary =
          segmentEnd + 1 < effective.length ? effective[segmentEnd + 1] : Number.POSITIVE_INFINITY;
        const leftRise = leftBoundary / Math.max(valley, 1e-12) > 1 + xi;
        const rightRise = rightBoundary / Math.max(valley, 1e-12) > 1 + xi;
        if (leftRise || rightRise) {
          for (let i = segmentStart; i <= segmentEnd; i += 1) {
            labels[ordering[i]] = clusterId;
          }
          clusterId += 1;
        }
      }
      segmentStart = null;
    }
  }

  if (clusterId === 0) {
    const fallbackThreshold = epsCutoff * (1 - xi);
    let fallbackStart: number | null = null;
    for (let pos = 0; pos <= effective.length; pos += 1) {
      const active =
        pos < effective.length &&
        Number.isFinite(effective[pos]) &&
        effective[pos] <= fallbackThreshold;
      if (active) {
        if (fallbackStart === null) {
          fallbackStart = pos;
        }
        continue;
      }
      if (fallbackStart !== null && pos - fallbackStart >= minSamples) {
        for (let i = fallbackStart; i < pos; i += 1) {
          labels[ordering[i]] = clusterId;
        }
        clusterId += 1;
      }
      fallbackStart = null;
    }
  }

  return labels;
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
  private xi: number;

  constructor(options: OPTICSOptions = {}) {
    this.minSamples = options.minSamples ?? 5;
    this.maxEps = options.maxEps ?? Number.POSITIVE_INFINITY;
    this.eps = options.eps;
    this.clusterMethod = options.clusterMethod ?? "dbscan";
    this.xi = options.xi ?? 0.05;

    if (!Number.isInteger(this.minSamples) || this.minSamples < 2) {
      throw new Error(`minSamples must be an integer >= 2. Got ${this.minSamples}.`);
    }
    if (!(Number.isFinite(this.maxEps) || this.maxEps === Number.POSITIVE_INFINITY) || this.maxEps <= 0) {
      throw new Error(`maxEps must be > 0 or Infinity. Got ${this.maxEps}.`);
    }
    if (this.eps !== undefined && (!Number.isFinite(this.eps) || this.eps <= 0)) {
      throw new Error(`eps must be finite and > 0 when provided. Got ${this.eps}.`);
    }
    if (!Number.isFinite(this.xi) || this.xi <= 0 || this.xi >= 1) {
      throw new Error(`xi must be finite and in (0, 1). Got ${this.xi}.`);
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
      const epsCutoff = chooseXiEps(reachability, coreDistances, this.maxEps);
      labels = extractXiClusters(ordering, reachability, coreDistances, this.minSamples, this.xi, epsCutoff);
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
