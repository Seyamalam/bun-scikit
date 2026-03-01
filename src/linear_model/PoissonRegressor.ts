import type { Matrix, RegressionModel, Vector } from "../types";
import { r2Score } from "../metrics/regression";
import { dot } from "../utils/linalg";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  validateRegressionInputs,
} from "../utils/validation";

export interface PoissonRegressorOptions {
  alpha?: number;
  fitIntercept?: boolean;
  maxIter?: number;
  tolerance?: number;
  learningRate?: number;
}

function safeExp(value: number): number {
  return Math.exp(Math.max(-50, Math.min(50, value)));
}

export class PoissonRegressor implements RegressionModel {
  coef_: Vector = [];
  intercept_ = 0;
  nIter_ = 0;

  private alpha: number;
  private fitIntercept: boolean;
  private maxIter: number;
  private tolerance: number;
  private learningRate: number;
  private fitted = false;

  constructor(options: PoissonRegressorOptions = {}) {
    this.alpha = options.alpha ?? 1;
    this.fitIntercept = options.fitIntercept ?? true;
    this.maxIter = options.maxIter ?? 1000;
    this.tolerance = options.tolerance ?? 1e-5;
    this.learningRate = options.learningRate ?? 0.01;
    this.validateOptions();
  }

  fit(X: Matrix, y: Vector, sampleWeight?: Vector): this {
    validateRegressionInputs(X, y);
    for (let i = 0; i < y.length; i += 1) {
      if (y[i] < 0) {
        throw new Error("PoissonRegressor requires non-negative targets.");
      }
    }

    const nSamples = X.length;
    const nFeatures = X[0].length;
    this.coef_ = new Array<number>(nFeatures).fill(0);
    this.intercept_ = 0;

    this.nIter_ = this.maxIter;
    for (let iter = 0; iter < this.maxIter; iter += 1) {
      const gradients = new Array<number>(nFeatures).fill(0);
      let interceptGrad = 0;

      for (let i = 0; i < nSamples; i += 1) {
        const eta = dot(X[i], this.coef_) + this.intercept_;
        const mu = safeExp(eta);
        const grad = mu - y[i];
        for (let f = 0; f < nFeatures; f += 1) {
          gradients[f] += grad * X[i][f];
        }
        if (this.fitIntercept) {
          interceptGrad += grad;
        }
      }

      let maxUpdate = 0;
      for (let f = 0; f < nFeatures; f += 1) {
        const g = gradients[f] / nSamples + this.alpha * this.coef_[f];
        const delta = this.learningRate * g;
        this.coef_[f] -= delta;
        maxUpdate = Math.max(maxUpdate, Math.abs(delta));
      }
      if (this.fitIntercept) {
        const delta = this.learningRate * (interceptGrad / nSamples);
        this.intercept_ -= delta;
        maxUpdate = Math.max(maxUpdate, Math.abs(delta));
      }

      if (maxUpdate <= this.tolerance) {
        this.nIter_ = iter + 1;
        break;
      }
    }

    this.fitted = true;
    return this;
  }

  predict(X: Matrix): Vector {
    this.assertFitted();
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.coef_.length) {
      throw new Error(`Feature size mismatch. Expected ${this.coef_.length}, got ${X[0].length}.`);
    }
    return X.map((row) => safeExp(dot(row, this.coef_) + this.intercept_));
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return r2Score(y, this.predict(X));
  }

  private validateOptions(): void {
    if (!Number.isFinite(this.alpha) || this.alpha < 0) {
      throw new Error(`alpha must be finite and >= 0. Got ${this.alpha}.`);
    }
    if (!Number.isInteger(this.maxIter) || this.maxIter < 1) {
      throw new Error(`maxIter must be an integer >= 1. Got ${this.maxIter}.`);
    }
    if (!Number.isFinite(this.tolerance) || this.tolerance < 0) {
      throw new Error(`tolerance must be finite and >= 0. Got ${this.tolerance}.`);
    }
    if (!Number.isFinite(this.learningRate) || this.learningRate <= 0) {
      throw new Error(`learningRate must be finite and > 0. Got ${this.learningRate}.`);
    }
  }

  private assertFitted(): void {
    if (!this.fitted) {
      throw new Error("PoissonRegressor has not been fitted.");
    }
  }
}