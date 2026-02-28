import type { Matrix, Vector } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";

export interface MiniBatchKMeansOptions {
  nClusters?: number;
  batchSize?: number;
  maxIter?: number;
  tolerance?: number;
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

function squaredDistance(a: Vector, b: Vector): number {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

function cloneMatrix(X: Matrix): Matrix {
  return X.map((row) => row.slice());
}

export class MiniBatchKMeans {
  clusterCenters_: Matrix | null = null;
  labels_: Vector | null = null;
  inertia_: number | null = null;
  nIter_: number | null = null;
  nFeaturesIn_: number | null = null;

  private nClusters: number;
  private batchSize: number;
  private maxIter: number;
  private tolerance: number;
  private randomState?: number;
  private counts_: Vector | null = null;
  private fitted = false;

  constructor(options: MiniBatchKMeansOptions = {}) {
    this.nClusters = options.nClusters ?? 8;
    this.batchSize = options.batchSize ?? 100;
    this.maxIter = options.maxIter ?? 300;
    this.tolerance = options.tolerance ?? 1e-3;
    this.randomState = options.randomState;

    if (!Number.isInteger(this.nClusters) || this.nClusters < 1) {
      throw new Error(`nClusters must be an integer >= 1. Got ${this.nClusters}.`);
    }
    if (!Number.isInteger(this.batchSize) || this.batchSize < 1) {
      throw new Error(`batchSize must be an integer >= 1. Got ${this.batchSize}.`);
    }
    if (!Number.isInteger(this.maxIter) || this.maxIter < 1) {
      throw new Error(`maxIter must be an integer >= 1. Got ${this.maxIter}.`);
    }
    if (!Number.isFinite(this.tolerance) || this.tolerance < 0) {
      throw new Error(`tolerance must be finite and >= 0. Got ${this.tolerance}.`);
    }
  }

  fit(X: Matrix): this {
    this.partialFit(X);
    const assignment = this.assignLabels(X, this.clusterCenters_!);
    this.labels_ = assignment.labels;
    this.inertia_ = assignment.inertia;
    return this;
  }

  partialFit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (this.nClusters > X.length && !this.fitted) {
      throw new Error(`nClusters (${this.nClusters}) cannot exceed sample count (${X.length}) on first fit.`);
    }

    const random =
      this.randomState === undefined
        ? Math.random
        : (() => {
            const rng = new Mulberry32(this.randomState);
            return () => rng.next();
          })();

    if (!this.clusterCenters_) {
      const indices = Array.from({ length: X.length }, (_, idx) => idx);
      for (let i = indices.length - 1; i > 0; i -= 1) {
        const j = Math.floor(random() * (i + 1));
        const tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
      }
      this.clusterCenters_ = indices.slice(0, this.nClusters).map((index) => X[index].slice());
      this.counts_ = new Array<number>(this.nClusters).fill(0);
      this.nFeaturesIn_ = X[0].length;
    } else if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }

    const batchSize = Math.min(this.batchSize, X.length);
    const toleranceSquared = this.tolerance * this.tolerance;
    let iterations = 0;

    for (let iter = 0; iter < this.maxIter; iter += 1) {
      iterations = iter + 1;
      const previousCenters = cloneMatrix(this.clusterCenters_);

      for (let i = 0; i < batchSize; i += 1) {
        const sampleIndex = Math.floor(random() * X.length);
        const sample = X[sampleIndex];
        let bestCluster = 0;
        let bestDistance = Number.POSITIVE_INFINITY;
        for (let c = 0; c < this.nClusters; c += 1) {
          const distance = squaredDistance(sample, this.clusterCenters_[c]);
          if (distance < bestDistance) {
            bestDistance = distance;
            bestCluster = c;
          }
        }

        this.counts_![bestCluster] += 1;
        const eta = 1 / this.counts_![bestCluster];
        for (let feature = 0; feature < this.nFeaturesIn_!; feature += 1) {
          this.clusterCenters_![bestCluster][feature] =
            (1 - eta) * this.clusterCenters_![bestCluster][feature] + eta * sample[feature];
        }
      }

      let maxShift = 0;
      for (let c = 0; c < this.nClusters; c += 1) {
        const shift = squaredDistance(previousCenters[c], this.clusterCenters_[c]);
        if (shift > maxShift) {
          maxShift = shift;
        }
      }
      if (maxShift <= toleranceSquared) {
        break;
      }
    }

    this.nIter_ = iterations;
    this.fitted = true;
    return this;
  }

  predict(X: Matrix): Vector {
    this.assertFitted();
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }
    return this.assignLabels(X, this.clusterCenters_!).labels;
  }

  fitPredict(X: Matrix): Vector {
    return this.fit(X).predict(X);
  }

  transform(X: Matrix): Matrix {
    this.assertFitted();
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }
    const out: Matrix = new Array(X.length);
    for (let i = 0; i < X.length; i += 1) {
      const row = new Array<number>(this.nClusters);
      for (let c = 0; c < this.nClusters; c += 1) {
        row[c] = Math.sqrt(squaredDistance(X[i], this.clusterCenters_![c]));
      }
      out[i] = row;
    }
    return out;
  }

  score(X: Matrix): number {
    this.assertFitted();
    return -this.assignLabels(X, this.clusterCenters_!).inertia;
  }

  private assignLabels(X: Matrix, centers: Matrix): { labels: Vector; inertia: number } {
    const labels = new Array<number>(X.length).fill(0);
    let inertia = 0;
    for (let i = 0; i < X.length; i += 1) {
      let bestCluster = 0;
      let bestDistance = Number.POSITIVE_INFINITY;
      for (let c = 0; c < centers.length; c += 1) {
        const distance = squaredDistance(X[i], centers[c]);
        if (distance < bestDistance) {
          bestDistance = distance;
          bestCluster = c;
        }
      }
      labels[i] = bestCluster;
      inertia += bestDistance;
    }
    return { labels, inertia };
  }

  private assertFitted(): void {
    if (!this.fitted || !this.clusterCenters_ || this.nFeaturesIn_ === null) {
      throw new Error("MiniBatchKMeans has not been fitted.");
    }
  }
}
