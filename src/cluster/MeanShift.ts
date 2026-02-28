import type { Matrix, Vector } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";

export interface MeanShiftOptions {
  bandwidth?: number;
  maxIter?: number;
  tolerance?: number;
  binSeeding?: boolean;
  minBinFreq?: number;
  clusterAll?: boolean;
}

function squaredDistance(a: Vector, b: Vector): number {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

function distance(a: Vector, b: Vector): number {
  return Math.sqrt(squaredDistance(a, b));
}

function estimateBandwidth(X: Matrix): number {
  const distances: number[] = [];
  for (let i = 0; i < X.length; i += 1) {
    let nearest = Number.POSITIVE_INFINITY;
    for (let j = 0; j < X.length; j += 1) {
      if (i === j) {
        continue;
      }
      const d = distance(X[i], X[j]);
      if (d < nearest) {
        nearest = d;
      }
    }
    if (Number.isFinite(nearest)) {
      distances.push(nearest);
    }
  }
  if (distances.length === 0) {
    return 1;
  }
  distances.sort((a, b) => a - b);
  return Math.max(1e-6, distances[Math.floor((distances.length - 1) * 0.6)]);
}

function roundedKey(row: Vector, bandwidth: number): string {
  const scale = Math.max(bandwidth, 1e-6);
  return row.map((value) => Math.round(value / scale)).join(",");
}

export class MeanShift {
  clusterCenters_: Matrix | null = null;
  labels_: Vector | null = null;
  nFeaturesIn_: number | null = null;

  private bandwidth?: number;
  private maxIter: number;
  private tolerance: number;
  private binSeeding: boolean;
  private minBinFreq: number;
  private clusterAll: boolean;
  private fitted = false;

  constructor(options: MeanShiftOptions = {}) {
    this.bandwidth = options.bandwidth;
    this.maxIter = options.maxIter ?? 300;
    this.tolerance = options.tolerance ?? 1e-3;
    this.binSeeding = options.binSeeding ?? false;
    this.minBinFreq = options.minBinFreq ?? 1;
    this.clusterAll = options.clusterAll ?? true;

    if (this.bandwidth !== undefined && (!Number.isFinite(this.bandwidth) || this.bandwidth <= 0)) {
      throw new Error(`bandwidth must be finite and > 0 when provided. Got ${this.bandwidth}.`);
    }
    if (!Number.isInteger(this.maxIter) || this.maxIter < 1) {
      throw new Error(`maxIter must be an integer >= 1. Got ${this.maxIter}.`);
    }
    if (!Number.isFinite(this.tolerance) || this.tolerance <= 0) {
      throw new Error(`tolerance must be finite and > 0. Got ${this.tolerance}.`);
    }
    if (!Number.isInteger(this.minBinFreq) || this.minBinFreq < 1) {
      throw new Error(`minBinFreq must be an integer >= 1. Got ${this.minBinFreq}.`);
    }
  }

  fit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    const bandwidth = this.bandwidth ?? estimateBandwidth(X);
    const toleranceSquared = this.tolerance * this.tolerance;
    const seeds = this.buildSeeds(X, bandwidth);
    const modes: Matrix = [];
    const supports: number[] = [];

    for (let seedIndex = 0; seedIndex < seeds.length; seedIndex += 1) {
      let current = seeds[seedIndex].slice();
      let support = 0;
      for (let iter = 0; iter < this.maxIter; iter += 1) {
        const inBand: number[] = [];
        for (let i = 0; i < X.length; i += 1) {
          if (distance(current, X[i]) <= bandwidth) {
            inBand.push(i);
          }
        }
        if (inBand.length === 0) {
          break;
        }
        support = inBand.length;
        const next = new Array<number>(X[0].length).fill(0);
        for (let i = 0; i < inBand.length; i += 1) {
          const point = X[inBand[i]];
          for (let feature = 0; feature < next.length; feature += 1) {
            next[feature] += point[feature];
          }
        }
        for (let feature = 0; feature < next.length; feature += 1) {
          next[feature] /= inBand.length;
        }
        if (squaredDistance(current, next) <= toleranceSquared) {
          current = next;
          break;
        }
        current = next;
      }
      modes.push(current);
      supports.push(support);
    }

    const order = Array.from({ length: modes.length }, (_, index) => index).sort(
      (a, b) => supports[b] - supports[a],
    );
    const centers: Matrix = [];
    for (let i = 0; i < order.length; i += 1) {
      const candidate = modes[order[i]];
      let duplicate = false;
      for (let j = 0; j < centers.length; j += 1) {
        if (distance(candidate, centers[j]) <= bandwidth * 0.5) {
          duplicate = true;
          break;
        }
      }
      if (!duplicate) {
        centers.push(candidate.slice());
      }
    }
    if (centers.length === 0) {
      centers.push(X[0].slice());
    }

    const labels = new Array<number>(X.length).fill(-1);
    for (let i = 0; i < X.length; i += 1) {
      let bestCluster = -1;
      let bestDistance = Number.POSITIVE_INFINITY;
      for (let cluster = 0; cluster < centers.length; cluster += 1) {
        const d = distance(X[i], centers[cluster]);
        if (d < bestDistance) {
          bestDistance = d;
          bestCluster = cluster;
        }
      }
      if (this.clusterAll || bestDistance <= bandwidth) {
        labels[i] = bestCluster;
      }
    }

    this.clusterCenters_ = centers;
    this.labels_ = labels;
    this.nFeaturesIn_ = X[0].length;
    this.fitted = true;
    return this;
  }

  fitPredict(X: Matrix): Vector {
    return this.fit(X).labels_!.slice();
  }

  predict(X: Matrix): Vector {
    this.assertFitted();
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }

    const out = new Array<number>(X.length).fill(-1);
    for (let i = 0; i < X.length; i += 1) {
      let best = 0;
      let bestDistance = Number.POSITIVE_INFINITY;
      for (let c = 0; c < this.clusterCenters_!.length; c += 1) {
        const d = distance(X[i], this.clusterCenters_![c]);
        if (d < bestDistance) {
          bestDistance = d;
          best = c;
        }
      }
      out[i] = best;
    }
    return out;
  }

  private buildSeeds(X: Matrix, bandwidth: number): Matrix {
    if (!this.binSeeding) {
      return X.map((row) => row.slice());
    }
    const bins = new Map<string, { count: number; sum: Vector }>();
    for (let i = 0; i < X.length; i += 1) {
      const key = roundedKey(X[i], bandwidth);
      const entry = bins.get(key);
      if (entry) {
        entry.count += 1;
        for (let feature = 0; feature < entry.sum.length; feature += 1) {
          entry.sum[feature] += X[i][feature];
        }
      } else {
        bins.set(key, { count: 1, sum: X[i].slice() });
      }
    }

    const seeds: Matrix = [];
    for (const value of bins.values()) {
      if (value.count < this.minBinFreq) {
        continue;
      }
      const center = new Array<number>(value.sum.length);
      for (let feature = 0; feature < value.sum.length; feature += 1) {
        center[feature] = value.sum[feature] / value.count;
      }
      seeds.push(center);
    }
    if (seeds.length === 0) {
      return X.map((row) => row.slice());
    }
    return seeds;
  }

  private assertFitted(): void {
    if (!this.fitted || !this.clusterCenters_ || this.nFeaturesIn_ === null) {
      throw new Error("MeanShift has not been fitted.");
    }
  }
}
