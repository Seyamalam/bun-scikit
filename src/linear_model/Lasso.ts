import type { Matrix, RegressionModel, Vector } from "../types";
import { r2Score } from "../metrics/regression";
import { dot } from "../utils/linalg";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  validateRegressionInputs,
} from "../utils/validation";
import { fitCoordinateDescent } from "./sharedRegularized";

export interface LassoOptions {
  alpha?: number;
  fitIntercept?: boolean;
  maxIter?: number;
  tolerance?: number;
}

export class Lasso implements RegressionModel {
  coef_: Vector = [];
  intercept_ = 0;
  nIter_ = 0;

  private alpha: number;
  private fitIntercept: boolean;
  private maxIter: number;
  private tolerance: number;
  private fitted = false;

  constructor(options: LassoOptions = {}) {
    this.alpha = options.alpha ?? 1;
    this.fitIntercept = options.fitIntercept ?? true;
    this.maxIter = options.maxIter ?? 1000;
    this.tolerance = options.tolerance ?? 1e-4;
    this.validateOptions();
  }

  fit(X: Matrix, y: Vector, sampleWeight?: Vector): this {
    validateRegressionInputs(X, y);
    const model = fitCoordinateDescent(
      X,
      y,
      this.alpha,
      1,
      this.fitIntercept,
      this.maxIter,
      this.tolerance,
    );
    this.coef_ = model.coef;
    this.intercept_ = model.intercept;
    this.nIter_ = model.nIter;
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
    if (!Number.isFinite(this.alpha) || this.alpha < 0) {
      throw new Error(`alpha must be finite and >= 0. Got ${this.alpha}.`);
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
      throw new Error("Lasso has not been fitted.");
    }
  }
}
