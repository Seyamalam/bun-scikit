import type { Matrix, RegressionModel, Vector } from "../types";
import { r2Score } from "../metrics/regression";
import { dot } from "../utils/linalg";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  validateRegressionInputs,
} from "../utils/validation";

export interface SGDRegressorOptions {
  fitIntercept?: boolean;
  learningRate?: number;
  maxIter?: number;
  tolerance?: number;
  l2?: number;
}

export class SGDRegressor implements RegressionModel {
  coef_: Vector = [];
  intercept_ = 0;

  private readonly fitIntercept: boolean;
  private readonly learningRate: number;
  private readonly maxIter: number;
  private readonly tolerance: number;
  private readonly l2: number;
  private isFitted = false;

  constructor(options: SGDRegressorOptions = {}) {
    this.fitIntercept = options.fitIntercept ?? true;
    this.learningRate = options.learningRate ?? 0.05;
    this.maxIter = options.maxIter ?? 10_000;
    this.tolerance = options.tolerance ?? 1e-6;
    this.l2 = options.l2 ?? 0;
  }

  fit(X: Matrix, y: Vector, sampleWeight?: Vector): this {
    validateRegressionInputs(X, y);
    const nSamples = X.length;
    const nFeatures = X[0].length;
    this.coef_ = new Array<number>(nFeatures).fill(0);
    this.intercept_ = 0;

    for (let iter = 0; iter < this.maxIter; iter += 1) {
      const gradients = new Array<number>(nFeatures).fill(0);
      let interceptGradient = 0;

      for (let i = 0; i < nSamples; i += 1) {
        const prediction = dot(X[i], this.coef_) + this.intercept_;
        const error = prediction - y[i];
        for (let j = 0; j < nFeatures; j += 1) {
          gradients[j] += error * X[i][j];
        }
        if (this.fitIntercept) {
          interceptGradient += error;
        }
      }

      let maxUpdate = 0;
      for (let j = 0; j < nFeatures; j += 1) {
        const grad = gradients[j] / nSamples + this.l2 * this.coef_[j];
        const delta = this.learningRate * grad;
        this.coef_[j] -= delta;
        const absDelta = Math.abs(delta);
        if (absDelta > maxUpdate) {
          maxUpdate = absDelta;
        }
      }
      if (this.fitIntercept) {
        const interceptDelta = this.learningRate * (interceptGradient / nSamples);
        this.intercept_ -= interceptDelta;
        const absInterceptDelta = Math.abs(interceptDelta);
        if (absInterceptDelta > maxUpdate) {
          maxUpdate = absInterceptDelta;
        }
      }

      if (maxUpdate < this.tolerance) {
        break;
      }
    }

    this.isFitted = true;
    return this;
  }

  predict(X: Matrix): Vector {
    if (!this.isFitted) {
      throw new Error("SGDRegressor has not been fitted.");
    }
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.coef_.length) {
      throw new Error(`Feature size mismatch. Expected ${this.coef_.length}, got ${X[0].length}.`);
    }
    return X.map((row) => dot(row, this.coef_) + this.intercept_);
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return r2Score(y, this.predict(X));
  }
}

