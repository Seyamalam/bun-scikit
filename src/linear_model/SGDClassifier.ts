import type { ClassificationModel, Matrix, Vector } from "../types";
import { accuracyScore } from "../metrics/classification";
import { dot } from "../utils/linalg";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  validateClassificationInputs,
} from "../utils/validation";

export type SGDClassifierLoss = "hinge" | "log_loss";

export interface SGDClassifierOptions {
  loss?: SGDClassifierLoss;
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

export class SGDClassifier implements ClassificationModel {
  coef_: Vector = [];
  intercept_ = 0;
  classes_: Vector = [0, 1];

  private readonly loss: SGDClassifierLoss;
  private readonly fitIntercept: boolean;
  private readonly learningRate: number;
  private readonly maxIter: number;
  private readonly tolerance: number;
  private readonly l2: number;
  private isFitted = false;

  constructor(options: SGDClassifierOptions = {}) {
    this.loss = options.loss ?? "hinge";
    this.fitIntercept = options.fitIntercept ?? true;
    this.learningRate = options.learningRate ?? 0.05;
    this.maxIter = options.maxIter ?? 10_000;
    this.tolerance = options.tolerance ?? 1e-6;
    this.l2 = options.l2 ?? 0;
  }

  fit(X: Matrix, y: Vector): this {
    validateClassificationInputs(X, y);
    const nSamples = X.length;
    const nFeatures = X[0].length;
    const ySigned = y.map((value) => (value === 1 ? 1 : -1));

    this.coef_ = new Array<number>(nFeatures).fill(0);
    this.intercept_ = 0;

    for (let iter = 0; iter < this.maxIter; iter += 1) {
      const gradients = new Array<number>(nFeatures).fill(0);
      let interceptGradient = 0;

      for (let i = 0; i < nSamples; i += 1) {
        const score = dot(X[i], this.coef_) + this.intercept_;

        if (this.loss === "hinge") {
          const margin = ySigned[i] * score;
          if (margin < 1) {
            const factor = -ySigned[i];
            for (let j = 0; j < nFeatures; j += 1) {
              gradients[j] += factor * X[i][j];
            }
            if (this.fitIntercept) {
              interceptGradient += factor;
            }
          }
        } else {
          const p = sigmoid(score);
          const error = p - y[i];
          for (let j = 0; j < nFeatures; j += 1) {
            gradients[j] += error * X[i][j];
          }
          if (this.fitIntercept) {
            interceptGradient += error;
          }
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

  predictProba(X: Matrix): Matrix {
    if (this.loss !== "log_loss") {
      throw new Error("predictProba is only available when loss='log_loss'.");
    }
    if (!this.isFitted) {
      throw new Error("SGDClassifier has not been fitted.");
    }
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.coef_.length) {
      throw new Error(`Feature size mismatch. Expected ${this.coef_.length}, got ${X[0].length}.`);
    }

    return X.map((row) => {
      const positive = sigmoid(dot(row, this.coef_) + this.intercept_);
      return [1 - positive, positive];
    });
  }

  predict(X: Matrix): Vector {
    if (!this.isFitted) {
      throw new Error("SGDClassifier has not been fitted.");
    }
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.coef_.length) {
      throw new Error(`Feature size mismatch. Expected ${this.coef_.length}, got ${X[0].length}.`);
    }

    if (this.loss === "log_loss") {
      return this.predictProba(X).map((pair) => (pair[1] >= 0.5 ? 1 : 0));
    }

    return X.map((row) => (dot(row, this.coef_) + this.intercept_ >= 0 ? 1 : 0));
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return accuracyScore(y, this.predict(X));
  }
}
