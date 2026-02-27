import type { Matrix, RegressionModel, Vector } from "../types";
import { r2Score } from "../metrics/regression";
import { dot } from "../utils/linalg";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  validateRegressionInputs,
} from "../utils/validation";

export interface LinearSVROptions {
  C?: number;
  epsilon?: number;
  fitIntercept?: boolean;
  learningRate?: number;
  maxIter?: number;
  tolerance?: number;
}

export class LinearSVR implements RegressionModel {
  coef_: Vector = [];
  intercept_ = 0;

  private C: number;
  private epsilon: number;
  private fitIntercept: boolean;
  private learningRate: number;
  private maxIter: number;
  private tolerance: number;
  private isFitted = false;

  constructor(options: LinearSVROptions = {}) {
    this.C = options.C ?? 1.0;
    this.epsilon = options.epsilon ?? 0.1;
    this.fitIntercept = options.fitIntercept ?? true;
    this.learningRate = options.learningRate ?? 0.05;
    this.maxIter = options.maxIter ?? 5000;
    this.tolerance = options.tolerance ?? 1e-6;
    this.validateOptions();
  }

  getParams(): LinearSVROptions {
    return {
      C: this.C,
      epsilon: this.epsilon,
      fitIntercept: this.fitIntercept,
      learningRate: this.learningRate,
      maxIter: this.maxIter,
      tolerance: this.tolerance,
    };
  }

  setParams(params: Partial<LinearSVROptions>): this {
    if (params.C !== undefined) this.C = params.C;
    if (params.epsilon !== undefined) this.epsilon = params.epsilon;
    if (params.fitIntercept !== undefined) this.fitIntercept = params.fitIntercept;
    if (params.learningRate !== undefined) this.learningRate = params.learningRate;
    if (params.maxIter !== undefined) this.maxIter = params.maxIter;
    if (params.tolerance !== undefined) this.tolerance = params.tolerance;
    this.validateOptions();
    return this;
  }

  fit(X: Matrix, y: Vector): this {
    validateRegressionInputs(X, y);
    const nSamples = X.length;
    const nFeatures = X[0].length;
    this.coef_ = new Array<number>(nFeatures).fill(0);
    this.intercept_ = 0;

    for (let iter = 0; iter < this.maxIter; iter += 1) {
      const gradients = new Array<number>(nFeatures).fill(0);
      let interceptGradient = 0;

      for (let i = 0; i < nSamples; i += 1) {
        const pred = dot(X[i], this.coef_) + this.intercept_;
        const residual = pred - y[i];
        if (Math.abs(residual) <= this.epsilon) {
          continue;
        }
        const sign = residual > 0 ? 1 : -1;
        for (let j = 0; j < nFeatures; j += 1) {
          gradients[j] += this.C * sign * X[i][j];
        }
        if (this.fitIntercept) {
          interceptGradient += this.C * sign;
        }
      }

      let maxUpdate = 0;
      for (let j = 0; j < nFeatures; j += 1) {
        const grad = gradients[j] / nSamples + this.coef_[j];
        const delta = this.learningRate * grad;
        this.coef_[j] -= delta;
        maxUpdate = Math.max(maxUpdate, Math.abs(delta));
      }
      if (this.fitIntercept) {
        const interceptDelta = this.learningRate * (interceptGradient / nSamples);
        this.intercept_ -= interceptDelta;
        maxUpdate = Math.max(maxUpdate, Math.abs(interceptDelta));
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
      throw new Error("LinearSVR has not been fitted.");
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

  private validateOptions(): void {
    if (!Number.isFinite(this.C) || this.C <= 0) {
      throw new Error(`C must be finite and > 0. Got ${this.C}.`);
    }
    if (!Number.isFinite(this.epsilon) || this.epsilon < 0) {
      throw new Error(`epsilon must be finite and >= 0. Got ${this.epsilon}.`);
    }
  }
}
