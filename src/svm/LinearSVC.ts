import type { ClassificationModel, Matrix, Vector } from "../types";
import { accuracyScore } from "../metrics/classification";
import { dot } from "../utils/linalg";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  validateClassificationInputs,
} from "../utils/validation";

export interface LinearSVCOptions {
  fitIntercept?: boolean;
  C?: number;
  learningRate?: number;
  maxIter?: number;
  tolerance?: number;
}

export class LinearSVC implements ClassificationModel {
  coef_: Vector = [];
  intercept_ = 0;
  classes_: Vector = [0, 1];

  private readonly fitIntercept: boolean;
  private readonly C: number;
  private readonly learningRate: number;
  private readonly maxIter: number;
  private readonly tolerance: number;
  private isFitted = false;

  constructor(options: LinearSVCOptions = {}) {
    this.fitIntercept = options.fitIntercept ?? true;
    this.C = options.C ?? 1.0;
    this.learningRate = options.learningRate ?? 0.05;
    this.maxIter = options.maxIter ?? 10_000;
    this.tolerance = options.tolerance ?? 1e-6;

    if (!Number.isFinite(this.C) || this.C <= 0) {
      throw new Error(`C must be > 0. Got ${this.C}.`);
    }
  }

  fit(X: Matrix, y: Vector): this {
    validateClassificationInputs(X, y);
    const nSamples = X.length;
    const nFeatures = X[0].length;

    this.coef_ = new Array<number>(nFeatures).fill(0);
    this.intercept_ = 0;
    const ySigned = y.map((value) => (value === 1 ? 1 : -1));

    for (let iter = 0; iter < this.maxIter; iter += 1) {
      const gradients = this.coef_.slice();
      let interceptGradient = 0;

      for (let i = 0; i < nSamples; i += 1) {
        const margin = ySigned[i] * (dot(X[i], this.coef_) + this.intercept_);
        if (margin < 1) {
          const factor = -this.C * ySigned[i];
          for (let j = 0; j < nFeatures; j += 1) {
            gradients[j] += factor * X[i][j];
          }
          if (this.fitIntercept) {
            interceptGradient += factor;
          }
        }
      }

      let maxUpdate = 0;
      for (let j = 0; j < nFeatures; j += 1) {
        const delta = this.learningRate * (gradients[j] / nSamples);
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

  decisionFunction(X: Matrix): Vector {
    if (!this.isFitted) {
      throw new Error("LinearSVC has not been fitted.");
    }
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.coef_.length) {
      throw new Error(`Feature size mismatch. Expected ${this.coef_.length}, got ${X[0].length}.`);
    }
    return X.map((row) => dot(row, this.coef_) + this.intercept_);
  }

  predict(X: Matrix): Vector {
    return this.decisionFunction(X).map((score) => (score >= 0 ? 1 : 0));
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return accuracyScore(y, this.predict(X));
  }
}
