import type { ClassificationModel, Matrix, Vector } from "../types";
import { accuracyScore } from "../metrics/classification";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  validateClassificationInputs,
} from "../utils/validation";
import { buildLabelIndex, uniqueSortedLabels } from "../utils/classification";

export type NearestCentroidMetric = "euclidean" | "manhattan";

export interface NearestCentroidOptions {
  metric?: NearestCentroidMetric;
  shrinkThreshold?: number | null;
}

function euclideanDistanceSquared(a: Vector, b: Vector): number {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

function manhattanDistance(a: Vector, b: Vector): number {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    sum += Math.abs(a[i] - b[i]);
  }
  return sum;
}

export class NearestCentroid implements ClassificationModel {
  classes_: Vector = [];
  centroids_: Matrix | null = null;
  nFeaturesIn_: number | null = null;

  private metric: NearestCentroidMetric;
  private shrinkThreshold: number | null;
  private labelToIndex: Map<number, number> = new Map();
  private fitted = false;

  constructor(options: NearestCentroidOptions = {}) {
    this.metric = options.metric ?? "euclidean";
    this.shrinkThreshold = options.shrinkThreshold ?? null;
    if (!(this.metric === "euclidean" || this.metric === "manhattan")) {
      throw new Error(`metric must be 'euclidean' or 'manhattan'. Got ${this.metric}.`);
    }
    if (this.shrinkThreshold !== null && (!Number.isFinite(this.shrinkThreshold) || this.shrinkThreshold < 0)) {
      throw new Error(`shrinkThreshold must be null or finite and >= 0. Got ${this.shrinkThreshold}.`);
    }
  }

  fit(X: Matrix, y: Vector): this {
    validateClassificationInputs(X, y);
    this.classes_ = uniqueSortedLabels(y);
    if (this.classes_.length < 2) {
      throw new Error("NearestCentroid requires at least two classes.");
    }
    this.labelToIndex = buildLabelIndex(this.classes_);

    const nFeatures = X[0].length;
    const centroids: Matrix = Array.from({ length: this.classes_.length }, () => new Array<number>(nFeatures).fill(0));
    const counts = new Array<number>(this.classes_.length).fill(0);

    for (let i = 0; i < X.length; i += 1) {
      const classIndex = this.labelToIndex.get(y[i]);
      if (classIndex === undefined) {
        throw new Error(`Unknown label '${y[i]}' in fit targets.`);
      }
      counts[classIndex] += 1;
      for (let j = 0; j < nFeatures; j += 1) {
        centroids[classIndex][j] += X[i][j];
      }
    }

    for (let c = 0; c < this.classes_.length; c += 1) {
      for (let j = 0; j < nFeatures; j += 1) {
        centroids[c][j] /= Math.max(1, counts[c]);
        if (this.shrinkThreshold !== null && Math.abs(centroids[c][j]) < this.shrinkThreshold) {
          centroids[c][j] = 0;
        }
      }
    }

    this.centroids_ = centroids;
    this.nFeaturesIn_ = nFeatures;
    this.fitted = true;
    return this;
  }

  predict(X: Matrix): Vector {
    this.assertFitted();
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }

    const predictions = new Array<number>(X.length).fill(0);
    for (let i = 0; i < X.length; i += 1) {
      let bestClass = 0;
      let bestDistance = Number.POSITIVE_INFINITY;
      for (let c = 0; c < this.classes_.length; c += 1) {
        const distance = this.metric === "manhattan"
          ? manhattanDistance(X[i], this.centroids_![c])
          : euclideanDistanceSquared(X[i], this.centroids_![c]);
        if (distance < bestDistance) {
          bestDistance = distance;
          bestClass = c;
        }
      }
      predictions[i] = this.classes_[bestClass];
    }
    return predictions;
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return accuracyScore(y, this.predict(X));
  }

  private assertFitted(): void {
    if (!this.fitted || !this.centroids_ || this.nFeaturesIn_ === null) {
      throw new Error("NearestCentroid has not been fitted.");
    }
  }
}

