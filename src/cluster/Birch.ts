import type { Matrix, Vector } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";
import { KMeans } from "./KMeans";

export interface BirchOptions {
  threshold?: number;
  branchingFactor?: number;
  nClusters?: number | null;
  computeLabels?: boolean;
}

interface MicroCluster {
  count: number;
  sum: Vector;
  center: Vector;
}

function squaredEuclideanDistance(a: Vector, b: Vector): number {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

function nearestIndex(centers: Matrix, sample: Vector): number {
  let bestIndex = 0;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (let i = 0; i < centers.length; i += 1) {
    const distance = squaredEuclideanDistance(sample, centers[i]);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = i;
    }
  }
  return bestIndex;
}

export class Birch {
  labels_: Vector | null = null;
  subclusterCenters_: Matrix | null = null;
  subclusterLabels_: Vector | null = null;
  clusterCenters_: Matrix | null = null;
  nFeaturesIn_: number | null = null;

  private threshold: number;
  private branchingFactor: number;
  private nClusters: number | null;
  private computeLabels: boolean;

  constructor(options: BirchOptions = {}) {
    this.threshold = options.threshold ?? 0.5;
    this.branchingFactor = options.branchingFactor ?? 50;
    this.nClusters = options.nClusters ?? 3;
    this.computeLabels = options.computeLabels ?? true;

    if (!Number.isFinite(this.threshold) || this.threshold <= 0) {
      throw new Error(`threshold must be finite and > 0. Got ${this.threshold}.`);
    }
    if (!Number.isInteger(this.branchingFactor) || this.branchingFactor < 2) {
      throw new Error(`branchingFactor must be an integer >= 2. Got ${this.branchingFactor}.`);
    }
    if (
      this.nClusters !== null &&
      (!Number.isInteger(this.nClusters) || this.nClusters < 1)
    ) {
      throw new Error(`nClusters must be null or an integer >= 1. Got ${this.nClusters}.`);
    }
  }

  fit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    const microClusters: MicroCluster[] = [];
    const thresholdSq = this.threshold * this.threshold;
    for (let i = 0; i < X.length; i += 1) {
      const sample = X[i];
      if (microClusters.length === 0) {
        microClusters.push({
          count: 1,
          sum: sample.slice(),
          center: sample.slice(),
        });
        continue;
      }

      let bestClusterIndex = 0;
      let bestDistance = Number.POSITIVE_INFINITY;
      for (let clusterIndex = 0; clusterIndex < microClusters.length; clusterIndex += 1) {
        const distance = squaredEuclideanDistance(sample, microClusters[clusterIndex].center);
        if (distance < bestDistance) {
          bestDistance = distance;
          bestClusterIndex = clusterIndex;
        }
      }

      const canCreateNewCluster =
        bestDistance > thresholdSq && microClusters.length < this.branchingFactor;
      if (canCreateNewCluster) {
        microClusters.push({
          count: 1,
          sum: sample.slice(),
          center: sample.slice(),
        });
      } else {
        const cluster = microClusters[bestClusterIndex];
        cluster.count += 1;
        for (let j = 0; j < sample.length; j += 1) {
          cluster.sum[j] += sample[j];
          cluster.center[j] = cluster.sum[j] / cluster.count;
        }
      }
    }

    const subclusterCenters = microClusters.map((cluster) => cluster.center.slice());
    this.subclusterCenters_ = subclusterCenters;
    this.nFeaturesIn_ = X[0].length;

    let subclusterLabels: Vector;
    let clusterCenters: Matrix;
    if (this.nClusters === null || this.nClusters >= subclusterCenters.length) {
      subclusterLabels = Array.from({ length: subclusterCenters.length }, (_, idx) => idx);
      clusterCenters = subclusterCenters.map((row) => row.slice());
    } else {
      const kmeans = new KMeans({
        nClusters: this.nClusters,
        nInit: 5,
        maxIter: 200,
        randomState: 0,
      }).fit(subclusterCenters);
      subclusterLabels = kmeans.labels_!.slice();
      clusterCenters = kmeans.clusterCenters_!.map((row) => row.slice());
    }

    this.subclusterLabels_ = subclusterLabels;
    this.clusterCenters_ = clusterCenters;

    if (this.computeLabels) {
      this.labels_ = X.map((sample) => {
        const subclusterIndex = nearestIndex(subclusterCenters, sample);
        return subclusterLabels[subclusterIndex];
      });
    } else {
      this.labels_ = null;
    }
    return this;
  }

  fitPredict(X: Matrix): Vector {
    this.fit(X);
    if (!this.labels_) {
      throw new Error("Birch was fitted with computeLabels=false; labels are unavailable.");
    }
    return this.labels_.slice();
  }

  predict(X: Matrix): Vector {
    this.assertFitted();
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }

    return X.map((sample) => {
      const subclusterIndex = nearestIndex(this.subclusterCenters_!, sample);
      return this.subclusterLabels_![subclusterIndex];
    });
  }

  transform(X: Matrix): Matrix {
    this.assertFitted();
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }

    return X.map((sample) =>
      this.subclusterCenters_!.map((center) => Math.sqrt(squaredEuclideanDistance(sample, center))),
    );
  }

  private assertFitted(): void {
    if (!this.subclusterCenters_ || !this.subclusterLabels_ || this.nFeaturesIn_ === null) {
      throw new Error("Birch has not been fitted.");
    }
  }
}
