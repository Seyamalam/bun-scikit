import type { Matrix, RegressionModel, Vector } from "../types";
import { r2Score } from "../metrics/regression";
import { dot } from "../utils/linalg";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  validateRegressionInputs,
} from "../utils/validation";

export type PassiveAggressiveRegressorLoss =
  | "epsilon_insensitive"
  | "squared_epsilon_insensitive";

export interface PassiveAggressiveRegressorOptions {
  C?: number;
  fitIntercept?: boolean;
  maxIter?: number;
  tolerance?: number;
  epsilon?: number;
  loss?: PassiveAggressiveRegressorLoss;
}

export class PassiveAggressiveRegressor implements RegressionModel {
  coef_: Vector = [];
  intercept_ = 0;
  nIter_ = 0;

  private C: number;
  private fitIntercept: boolean;
  private maxIter: number;
  private tolerance: number;
  private epsilon: number;
  private loss: PassiveAggressiveRegressorLoss;
  private fitted = false;

  constructor(options: PassiveAggressiveRegressorOptions = {}) {
    this.C = options.C ?? 1;
    this.fitIntercept = options.fitIntercept ?? true;
    this.maxIter = options.maxIter ?? 1000;
    this.tolerance = options.tolerance ?? 1e-4;
    this.epsilon = options.epsilon ?? 0.1;
    this.loss = options.loss ?? "epsilon_insensitive";
    this.validateOptions();
  }

  fit(X: Matrix, y: Vector, sampleWeight?: Vector): this {
    validateRegressionInputs(X, y);
    const nFeatures = X[0].length;
    this.coef_ = new Array<number>(nFeatures).fill(0);
    this.intercept_ = 0;

    this.nIter_ = this.maxIter;
    for (let iter = 0; iter < this.maxIter; iter += 1) {
      let maxUpdate = 0;
      for (let i = 0; i < X.length; i += 1) {
        const pred = dot(X[i], this.coef_) + this.intercept_;
        const residual = pred - y[i];
        const lossValue = Math.max(0, Math.abs(residual) - this.epsilon);
        if (lossValue <= 0) {
          continue;
        }

        let normSq = 0;
        for (let f = 0; f < nFeatures; f += 1) {
          normSq += X[i][f] * X[i][f];
        }
        if (this.fitIntercept) {
          normSq += 1;
        }

        let tau: number;
        if (this.loss === "squared_epsilon_insensitive") {
          tau = lossValue / Math.max(normSq + 1 / (2 * this.C), 1e-12);
        } else {
          tau = Math.min(this.C, lossValue / Math.max(normSq, 1e-12));
        }
        const direction = residual >= 0 ? 1 : -1;

        for (let f = 0; f < nFeatures; f += 1) {
          const delta = tau * direction * X[i][f];
          this.coef_[f] -= delta;
          maxUpdate = Math.max(maxUpdate, Math.abs(delta));
        }
        if (this.fitIntercept) {
          this.intercept_ -= tau * direction;
          maxUpdate = Math.max(maxUpdate, Math.abs(tau));
        }
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
    if (!Number.isInteger(this.maxIter) || this.maxIter < 1) {
      throw new Error(`maxIter must be an integer >= 1. Got ${this.maxIter}.`);
    }
    if (!Number.isFinite(this.tolerance) || this.tolerance < 0) {
      throw new Error(`tolerance must be finite and >= 0. Got ${this.tolerance}.`);
    }
  }

  private assertFitted(): void {
    if (!this.fitted) {
      throw new Error("PassiveAggressiveRegressor has not been fitted.");
    }
  }
}