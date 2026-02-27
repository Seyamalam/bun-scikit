import type { Matrix, Vector } from "../types";
import { assertConsistentRowSize, assertNonEmptyMatrix } from "../utils/validation";

export interface KNNImputerOptions {
  nNeighbors?: number;
  weights?: "uniform" | "distance";
}

function isMissing(value: number): boolean {
  return Number.isNaN(value);
}

function assertFiniteOrMissing(X: Matrix, label = "X"): void {
  for (let i = 0; i < X.length; i += 1) {
    for (let j = 0; j < X[i].length; j += 1) {
      const value = X[i][j];
      if (!Number.isFinite(value) && !isMissing(value)) {
        throw new Error(`${label} contains non-finite non-missing value at [${i}, ${j}].`);
      }
    }
  }
}

interface NeighborDistance {
  index: number;
  distance: number;
}

export class KNNImputer {
  statistics_: Vector | null = null;

  private readonly nNeighbors: number;
  private readonly weights: "uniform" | "distance";
  private fitX: Matrix | null = null;
  private nFeatures = 0;

  constructor(options: KNNImputerOptions = {}) {
    this.nNeighbors = options.nNeighbors ?? 5;
    this.weights = options.weights ?? "uniform";
    if (!Number.isInteger(this.nNeighbors) || this.nNeighbors < 1) {
      throw new Error(`nNeighbors must be an integer >= 1. Got ${this.nNeighbors}.`);
    }
  }

  fit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteOrMissing(X);

    this.nFeatures = X[0].length;
    const stats = new Array<number>(this.nFeatures);
    for (let feature = 0; feature < this.nFeatures; feature += 1) {
      let sum = 0;
      let count = 0;
      for (let i = 0; i < X.length; i += 1) {
        const value = X[i][feature];
        if (!isMissing(value)) {
          sum += value;
          count += 1;
        }
      }
      if (count === 0) {
        throw new Error(
          `Feature at index ${feature} has only missing values and cannot be imputed by KNNImputer.`,
        );
      }
      stats[feature] = sum / count;
    }

    this.statistics_ = stats;
    this.fitX = X.map((row) => row.slice());
    return this;
  }

  transform(X: Matrix): Matrix {
    if (this.fitX === null || this.statistics_ === null) {
      throw new Error("KNNImputer has not been fitted.");
    }
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteOrMissing(X);
    if (X[0].length !== this.nFeatures) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeatures}, got ${X[0].length}.`);
    }

    const out = X.map((row) => row.slice());
    for (let i = 0; i < out.length; i += 1) {
      const row = out[i];
      let hasMissing = false;
      for (let j = 0; j < row.length; j += 1) {
        if (isMissing(row[j])) {
          hasMissing = true;
          break;
        }
      }
      if (!hasMissing) {
        continue;
      }

      const neighbors = this.computeNeighborDistances(row);
      for (let feature = 0; feature < row.length; feature += 1) {
        if (!isMissing(row[feature])) {
          continue;
        }
        row[feature] = this.imputeFeatureFromNeighbors(neighbors, feature);
      }
    }

    return out;
  }

  fitTransform(X: Matrix): Matrix {
    return this.fit(X).transform(X);
  }

  private computeNeighborDistances(row: Vector): NeighborDistance[] {
    const distances: NeighborDistance[] = [];
    for (let i = 0; i < this.fitX!.length; i += 1) {
      const other = this.fitX![i];
      let sumSquared = 0;
      let overlap = 0;
      for (let feature = 0; feature < this.nFeatures; feature += 1) {
        const a = row[feature];
        const b = other[feature];
        if (isMissing(a) || isMissing(b)) {
          continue;
        }
        const diff = a - b;
        sumSquared += diff * diff;
        overlap += 1;
      }
      if (overlap === 0) {
        continue;
      }
      distances.push({
        index: i,
        distance: Math.sqrt(sumSquared),
      });
    }
    distances.sort((a, b) => a.distance - b.distance);
    return distances;
  }

  private imputeFeatureFromNeighbors(
    neighbors: NeighborDistance[],
    feature: number,
  ): number {
    let weightedSum = 0;
    let weightSum = 0;
    let used = 0;
    for (let i = 0; i < neighbors.length && used < this.nNeighbors; i += 1) {
      const neighbor = neighbors[i];
      const value = this.fitX![neighbor.index][feature];
      if (isMissing(value)) {
        continue;
      }
      const weight =
        this.weights === "distance"
          ? 1 / Math.max(neighbor.distance, 1e-12)
          : 1;
      weightedSum += weight * value;
      weightSum += weight;
      used += 1;
    }

    if (weightSum === 0) {
      return this.statistics_![feature];
    }
    return weightedSum / weightSum;
  }
}
