import type { ClassificationModel, Matrix, Vector } from "../types";
import { dot } from "../utils/linalg";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  validateClassificationInputs,
} from "../utils/validation";
import { accuracyScore } from "../metrics/classification";

export interface LogisticRegressionOptions {
  fitIntercept?: boolean;
  learningRate?: number;
  maxIter?: number;
  tolerance?: number;
  l2?: number;
}

function sigmoid(z: number): number {
  if (z >= 0) {
    const expNeg = Math.exp(-z);
    return 1 / (1 + expNeg);
  }
  const expPos = Math.exp(z);
  return expPos / (1 + expPos);
}

export class LogisticRegression implements ClassificationModel {
  coef_: Vector = [];
  intercept_ = 0;
  classes_: Vector = [0, 1];

  private readonly fitIntercept: boolean;
  private readonly learningRate: number;
  private readonly maxIter: number;
  private readonly tolerance: number;
  private readonly l2: number;
  private isFitted = false;

  constructor(options: LogisticRegressionOptions = {}) {
    this.fitIntercept = options.fitIntercept ?? true;
    this.learningRate = options.learningRate ?? 0.1;
    this.maxIter = options.maxIter ?? 20_000;
    this.tolerance = options.tolerance ?? 1e-8;
    this.l2 = options.l2 ?? 0;
  }

  fit(X: Matrix, y: Vector): this {
    validateClassificationInputs(X, y);

    const nSamples = X.length;
    const nFeatures = X[0].length;
    this.coef_ = new Array(nFeatures).fill(0);
    this.intercept_ = 0;
    const gradients = new Array(nFeatures).fill(0);

    for (let iter = 0; iter < this.maxIter; iter += 1) {
      let interceptGradient = 0;
      for (let j = 0; j < nFeatures; j += 1) {
        gradients[j] = 0;
      }

      for (let i = 0; i < nSamples; i += 1) {
        const row = X[i];
        const z = this.intercept_ + dot(row, this.coef_);
        const prediction = sigmoid(z);
        const target = y[i];
        const error = prediction - target;
        interceptGradient += error;

        for (let j = 0; j < nFeatures; j += 1) {
          gradients[j] += error * row[j];
        }
      }

      const scale = 1 / nSamples;
      let maxUpdate = 0;
      for (let j = 0; j < nFeatures; j += 1) {
        const l2Term = this.l2 > 0 ? this.l2 * this.coef_[j] : 0;
        const delta = this.learningRate * (scale * gradients[j] + scale * l2Term);
        this.coef_[j] -= delta;
        const absDelta = Math.abs(delta);
        if (absDelta > maxUpdate) {
          maxUpdate = absDelta;
        }
      }

      if (this.fitIntercept) {
        const interceptDelta = this.learningRate * scale * interceptGradient;
        this.intercept_ -= interceptDelta;
        const absInterceptDelta = Math.abs(interceptDelta);
        if (absInterceptDelta > maxUpdate) {
          maxUpdate = absInterceptDelta;
        }
      }

      if (maxUpdate < this.tolerance) {
        this.isFitted = true;
        return this;
      }
    }

    this.isFitted = true;
    return this;
  }

  predictProba(X: Matrix): Matrix {
    if (!this.isFitted) {
      throw new Error("LogisticRegression has not been fitted.");
    }

    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    if (X[0].length !== this.coef_.length) {
      throw new Error(
        `Feature size mismatch. Expected ${this.coef_.length}, got ${X[0].length}.`,
      );
    }

    return X.map((row) => {
      const positive = sigmoid(this.intercept_ + dot(row, this.coef_));
      return [1 - positive, positive];
    });
  }

  predict(X: Matrix): Vector {
    return this.predictProba(X).map((pair) => (pair[1] >= 0.5 ? 1 : 0));
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return accuracyScore(y, this.predict(X));
  }
}
