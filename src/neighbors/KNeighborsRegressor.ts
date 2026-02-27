import type { Matrix, RegressionModel, Vector } from "../types";
import { r2Score } from "../metrics/regression";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  validateRegressionInputs,
} from "../utils/validation";

export type KNeighborsRegressorWeights = "uniform" | "distance";

export interface KNeighborsRegressorOptions {
  nNeighbors?: number;
  weights?: KNeighborsRegressorWeights;
}

function squaredEuclideanDistance(a: Vector, b: Vector): number {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

export class KNeighborsRegressor implements RegressionModel {
  private nNeighbors: number;
  private weights: KNeighborsRegressorWeights;
  private XTrain: Matrix | null = null;
  private yTrain: Vector | null = null;

  constructor(options: KNeighborsRegressorOptions = {}) {
    const nNeighbors = options.nNeighbors ?? 5;
    if (!Number.isInteger(nNeighbors) || nNeighbors < 1) {
      throw new Error(`nNeighbors must be a positive integer. Got ${nNeighbors}.`);
    }
    const weights = options.weights ?? "uniform";
    if (!(weights === "uniform" || weights === "distance")) {
      throw new Error(`weights must be 'uniform' or 'distance'. Got ${weights}.`);
    }
    this.nNeighbors = nNeighbors;
    this.weights = weights;
  }

  fit(X: Matrix, y: Vector, sampleWeight?: Vector): this {
    validateRegressionInputs(X, y);
    if (this.nNeighbors > X.length) {
      throw new Error(
        `nNeighbors (${this.nNeighbors}) cannot exceed training size (${X.length}).`,
      );
    }
    this.XTrain = X.map((row) => row.slice());
    this.yTrain = y.slice();
    return this;
  }

  predict(X: Matrix): Vector {
    if (!this.XTrain || !this.yTrain) {
      throw new Error("KNeighborsRegressor has not been fitted.");
    }

    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.XTrain[0].length) {
      throw new Error(
        `Feature size mismatch. Expected ${this.XTrain[0].length}, got ${X[0].length}.`,
      );
    }

    return X.map((sample) => {
      const distances = this.XTrain!.map((row, idx) => ({
        distance: Math.sqrt(squaredEuclideanDistance(sample, row)),
        target: this.yTrain![idx],
      }));
      distances.sort((a, b) => a.distance - b.distance);
      const neighbors = distances.slice(0, this.nNeighbors);

      if (this.weights === "uniform") {
        let sum = 0;
        for (let i = 0; i < neighbors.length; i += 1) {
          sum += neighbors[i].target;
        }
        return sum / neighbors.length;
      }

      let weightedSum = 0;
      let weightTotal = 0;
      for (let i = 0; i < neighbors.length; i += 1) {
        const distance = neighbors[i].distance;
        if (distance === 0) {
          return neighbors[i].target;
        }
        const weight = 1 / Math.max(distance, 1e-12);
        weightedSum += weight * neighbors[i].target;
        weightTotal += weight;
      }
      return weightTotal === 0 ? 0 : weightedSum / weightTotal;
    });
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return r2Score(y, this.predict(X)) as number;
  }
}
