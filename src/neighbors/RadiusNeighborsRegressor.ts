import type { Matrix, RegressionModel, Vector } from "../types";
import { r2Score } from "../metrics/regression";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  validateRegressionInputs,
} from "../utils/validation";

export type RadiusNeighborsRegressorWeights = "uniform" | "distance";

export interface RadiusNeighborsRegressorOptions {
  radius?: number;
  weights?: RadiusNeighborsRegressorWeights;
}

function euclideanDistance(a: Vector, b: Vector): number {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

export class RadiusNeighborsRegressor implements RegressionModel {
  private radius: number;
  private weights: RadiusNeighborsRegressorWeights;
  private XTrain: Matrix | null = null;
  private yTrain: Vector | null = null;

  constructor(options: RadiusNeighborsRegressorOptions = {}) {
    this.radius = options.radius ?? 1;
    this.weights = options.weights ?? "uniform";
    if (!Number.isFinite(this.radius) || this.radius <= 0) {
      throw new Error(`radius must be finite and > 0. Got ${this.radius}.`);
    }
    if (!(this.weights === "uniform" || this.weights === "distance")) {
      throw new Error(`weights must be 'uniform' or 'distance'. Got ${this.weights}.`);
    }
  }

  fit(X: Matrix, y: Vector, sampleWeight?: Vector): this {
    validateRegressionInputs(X, y);
    this.XTrain = X.map((row) => row.slice());
    this.yTrain = y.slice();
    return this;
  }

  predict(X: Matrix): Vector {
    this.assertFitted();
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.XTrain![0].length) {
      throw new Error(`Feature size mismatch. Expected ${this.XTrain![0].length}, got ${X[0].length}.`);
    }

    return X.map((sample) => {
      const neighbors: Array<{ distance: number; target: number }> = [];
      for (let i = 0; i < this.XTrain!.length; i += 1) {
        const distance = euclideanDistance(sample, this.XTrain![i]);
        if (distance <= this.radius) {
          neighbors.push({ distance, target: this.yTrain![i] });
        }
      }
      if (neighbors.length === 0) {
        return Number.NaN;
      }

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
      return weightTotal <= 0 ? Number.NaN : weightedSum / weightTotal;
    });
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    const pred = this.predict(X);
    for (let i = 0; i < pred.length; i += 1) {
      if (!Number.isFinite(pred[i])) {
        return Number.NaN;
      }
    }
    return r2Score(y, pred);
  }

  private assertFitted(): void {
    if (!this.XTrain || !this.yTrain) {
      throw new Error("RadiusNeighborsRegressor has not been fitted.");
    }
  }
}
