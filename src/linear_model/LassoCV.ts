import type { Matrix, RegressionModel, Vector } from "../types";
import { r2Score } from "../metrics/regression";
import { dot } from "../utils/linalg";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  validateRegressionInputs,
} from "../utils/validation";
import { Lasso } from "./Lasso";
import { crossValidatedMse } from "./sharedRegularizedCV";

export interface LassoCVOptions {
  alphas?: number[];
  cv?: number;
  fitIntercept?: boolean;
  maxIter?: number;
  tolerance?: number;
  randomState?: number;
}

export class LassoCV implements RegressionModel {
  alpha_ = 1;
  coef_: Vector = [];
  intercept_ = 0;
  msePath_: Vector = [];

  private alphas: number[];
  private cv: number;
  private fitIntercept: boolean;
  private maxIter: number;
  private tolerance: number;
  private randomState?: number;
  private fitted = false;

  constructor(options: LassoCVOptions = {}) {
    this.alphas = options.alphas ?? [1e-4, 1e-3, 1e-2, 1e-1, 1, 10];
    this.cv = options.cv ?? 5;
    this.fitIntercept = options.fitIntercept ?? true;
    this.maxIter = options.maxIter ?? 1000;
    this.tolerance = options.tolerance ?? 1e-4;
    this.randomState = options.randomState;
    this.validateOptions();
  }

  fit(X: Matrix, y: Vector, sampleWeight?: Vector): this {
    validateRegressionInputs(X, y);

    let bestAlpha = this.alphas[0];
    let bestMse = Number.POSITIVE_INFINITY;
    this.msePath_ = new Array<number>(this.alphas.length).fill(0);

    for (let i = 0; i < this.alphas.length; i += 1) {
      const alpha = this.alphas[i];
      const mse = crossValidatedMse(
        () =>
          new Lasso({
            alpha,
            fitIntercept: this.fitIntercept,
            maxIter: this.maxIter,
            tolerance: this.tolerance,
          }),
        X,
        y,
        { cv: this.cv, randomState: this.randomState },
      );
      this.msePath_[i] = mse;
      if (mse < bestMse) {
        bestMse = mse;
        bestAlpha = alpha;
      }
    }

    const finalModel = new Lasso({
      alpha: bestAlpha,
      fitIntercept: this.fitIntercept,
      maxIter: this.maxIter,
      tolerance: this.tolerance,
    }).fit(X, y);
    this.alpha_ = bestAlpha;
    this.coef_ = finalModel.coef_.slice();
    this.intercept_ = finalModel.intercept_;
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
    return X.map((row) => this.intercept_ + dot(row, this.coef_));
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return r2Score(y, this.predict(X));
  }

  private validateOptions(): void {
    if (!Number.isInteger(this.cv) || this.cv < 2) {
      throw new Error(`cv must be an integer >= 2. Got ${this.cv}.`);
    }
    if (!Number.isInteger(this.maxIter) || this.maxIter < 1) {
      throw new Error(`maxIter must be an integer >= 1. Got ${this.maxIter}.`);
    }
    if (!Number.isFinite(this.tolerance) || this.tolerance < 0) {
      throw new Error(`tolerance must be finite and >= 0. Got ${this.tolerance}.`);
    }
    if (!Array.isArray(this.alphas) || this.alphas.length === 0) {
      throw new Error("alphas must be a non-empty numeric array.");
    }
    for (let i = 0; i < this.alphas.length; i += 1) {
      if (!Number.isFinite(this.alphas[i]) || this.alphas[i] < 0) {
        throw new Error(`alphas must contain finite values >= 0. Got ${this.alphas[i]}.`);
      }
    }
  }

  private assertFitted(): void {
    if (!this.fitted) {
      throw new Error("LassoCV has not been fitted.");
    }
  }
}
