import type { Matrix, Vector } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";

export interface KMeansOptions {
  nClusters?: number;
  nInit?: number;
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

interface AssignmentResult {
  labels: Vector;
  inertia: number;
}

function squaredEuclideanDistance(a: Vector, b: Vector): number {
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

function randomInt(random: () => number, maxExclusive: number): number {
  const value = Math.floor(random() * maxExclusive);
  return Math.min(maxExclusive - 1, Math.max(0, value));
}

export class KMeans {
  clusterCenters_: Matrix | null = null;
  labels_: Vector | null = null;
  inertia_: number | null = null;
  nIter_: number | null = null;
  nFeaturesIn_: number | null = null;

  private readonly nClusters: number;
  private readonly nInit: number;
  private readonly maxIter: number;
  private readonly tolerance: number;
  private readonly randomState?: number;
  private isFitted = false;

  constructor(options: KMeansOptions = {}) {
    this.nClusters = options.nClusters ?? 8;
    this.nInit = options.nInit ?? 10;
    this.maxIter = options.maxIter ?? 300;
    this.tolerance = options.tolerance ?? 1e-4;
    this.randomState = options.randomState;

    if (!Number.isInteger(this.nClusters) || this.nClusters < 1) {
      throw new Error(`nClusters must be an integer >= 1. Got ${this.nClusters}.`);
    }
    if (!Number.isInteger(this.nInit) || this.nInit < 1) {
      throw new Error(`nInit must be an integer >= 1. Got ${this.nInit}.`);
    }
    if (!Number.isInteger(this.maxIter) || this.maxIter < 1) {
      throw new Error(`maxIter must be an integer >= 1. Got ${this.maxIter}.`);
    }
    if (!Number.isFinite(this.tolerance) || this.tolerance < 0) {
      throw new Error(`tolerance must be finite and >= 0. Got ${this.tolerance}.`);
    }
  }

  fit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    const nSamples = X.length;
    const nFeatures = X[0].length;
    if (this.nClusters > nSamples) {
      throw new Error(
        `nClusters (${this.nClusters}) cannot exceed sample count (${nSamples}).`,
      );
    }

    let bestCenters: Matrix | null = null;
    let bestLabels: Vector | null = null;
    let bestInertia = Number.POSITIVE_INFINITY;
    let bestNIter = 0;

    for (let initIndex = 0; initIndex < this.nInit; initIndex += 1) {
      let random: () => number;
      if (this.randomState === undefined) {
        random = Math.random;
      } else {
        const rng = new Mulberry32(this.randomState + initIndex * 104_729);
        random = () => rng.next();
      }
      const initialCenters = this.initializeCenters(X, random);
      const run = this.runSingleInitialization(X, initialCenters, random);

      if (run.inertia < bestInertia) {
        bestInertia = run.inertia;
        bestCenters = cloneMatrix(run.clusterCenters);
        bestLabels = run.labels.slice();
        bestNIter = run.nIter;
      }
    }

    this.clusterCenters_ = bestCenters;
    this.labels_ = bestLabels;
    this.inertia_ = bestInertia;
    this.nIter_ = bestNIter;
    this.nFeaturesIn_ = nFeatures;
    this.isFitted = true;
    return this;
  }

  predict(X: Matrix): Vector {
    const centers = this.getFittedCenters();
    this.validatePredictionInput(X, centers[0].length);
    return this.assignLabels(X, centers).labels;
  }

  fitPredict(X: Matrix): Vector {
    this.fit(X);
    return this.labels_!.slice();
  }

  transform(X: Matrix): Matrix {
    const centers = this.getFittedCenters();
    this.validatePredictionInput(X, centers[0].length);

    const distances: Matrix = new Array(X.length);
    for (let rowIndex = 0; rowIndex < X.length; rowIndex += 1) {
      const row = X[rowIndex];
      const outRow = new Array<number>(centers.length);
      for (let clusterIndex = 0; clusterIndex < centers.length; clusterIndex += 1) {
        outRow[clusterIndex] = Math.sqrt(
          squaredEuclideanDistance(row, centers[clusterIndex]),
        );
      }
      distances[rowIndex] = outRow;
    }

    return distances;
  }

  score(X: Matrix): number {
    const centers = this.getFittedCenters();
    this.validatePredictionInput(X, centers[0].length);
    return -this.assignLabels(X, centers).inertia;
  }

  private getFittedCenters(): Matrix {
    if (!this.isFitted || !this.clusterCenters_) {
      throw new Error("KMeans has not been fitted.");
    }
    return this.clusterCenters_;
  }

  private validatePredictionInput(X: Matrix, expectedFeatures: number): void {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== expectedFeatures) {
      throw new Error(`Feature size mismatch. Expected ${expectedFeatures}, got ${X[0].length}.`);
    }
  }

  private initializeCenters(X: Matrix, random: () => number): Matrix {
    const indices = Array.from({ length: X.length }, (_, idx) => idx);
    for (let i = indices.length - 1; i > 0; i -= 1) {
      const j = randomInt(random, i + 1);
      const tmp = indices[i];
      indices[i] = indices[j];
      indices[j] = tmp;
    }
    const selected = indices.slice(0, this.nClusters);
    return selected.map((sampleIndex) => X[sampleIndex].slice());
  }

  private runSingleInitialization(
    X: Matrix,
    initialCenters: Matrix,
    random: () => number,
  ): {
    clusterCenters: Matrix;
    labels: Vector;
    inertia: number;
    nIter: number;
  } {
    let centers = cloneMatrix(initialCenters);
    let nIter = 0;
    const toleranceSquared = this.tolerance * this.tolerance;

    for (let iter = 0; iter < this.maxIter; iter += 1) {
      nIter = iter + 1;
      const assignment = this.assignLabels(X, centers);
      const updatedCenters = this.updateCenters(X, assignment.labels, random);

      let maxShift = 0;
      for (let clusterIndex = 0; clusterIndex < this.nClusters; clusterIndex += 1) {
        const shift = squaredEuclideanDistance(centers[clusterIndex], updatedCenters[clusterIndex]);
        if (shift > maxShift) {
          maxShift = shift;
        }
      }

      centers = updatedCenters;
      if (maxShift <= toleranceSquared) {
        break;
      }
    }

    const finalAssignment = this.assignLabels(X, centers);
    return {
      clusterCenters: centers,
      labels: finalAssignment.labels,
      inertia: finalAssignment.inertia,
      nIter,
    };
  }

  private assignLabels(X: Matrix, centers: Matrix): AssignmentResult {
    const labels = new Array<number>(X.length).fill(0);
    let inertia = 0;

    for (let rowIndex = 0; rowIndex < X.length; rowIndex += 1) {
      const row = X[rowIndex];
      let bestCluster = 0;
      let bestDistance = Number.POSITIVE_INFINITY;
      for (let clusterIndex = 0; clusterIndex < centers.length; clusterIndex += 1) {
        const distance = squaredEuclideanDistance(row, centers[clusterIndex]);
        if (distance < bestDistance) {
          bestDistance = distance;
          bestCluster = clusterIndex;
        }
      }
      labels[rowIndex] = bestCluster;
      inertia += bestDistance;
    }

    return { labels, inertia };
  }

  private updateCenters(
    X: Matrix,
    labels: Vector,
    random: () => number,
  ): Matrix {
    const nFeatures = X[0].length;
    const centers: Matrix = Array.from({ length: this.nClusters }, () =>
      new Array(nFeatures).fill(0),
    );
    const counts = new Array<number>(this.nClusters).fill(0);

    for (let sampleIndex = 0; sampleIndex < X.length; sampleIndex += 1) {
      const clusterIndex = labels[sampleIndex];
      counts[clusterIndex] += 1;
      const row = X[sampleIndex];
      for (let featureIndex = 0; featureIndex < nFeatures; featureIndex += 1) {
        centers[clusterIndex][featureIndex] += row[featureIndex];
      }
    }

    for (let clusterIndex = 0; clusterIndex < this.nClusters; clusterIndex += 1) {
      const count = counts[clusterIndex];
      if (count === 0) {
        const fallback = X[randomInt(random, X.length)];
        centers[clusterIndex] = fallback.slice();
        continue;
      }

      for (let featureIndex = 0; featureIndex < nFeatures; featureIndex += 1) {
        centers[clusterIndex][featureIndex] /= count;
      }
    }

    return centers;
  }
}
