import type { Matrix, Vector } from "../types";
import { r2Score } from "../metrics/regression";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
  validateRegressionInputs,
} from "../utils/validation";
import { Lasso } from "./Lasso";

export interface MultiTaskLassoOptions {
  alpha?: number;
  fitIntercept?: boolean;
  maxIter?: number;
  tolerance?: number;
}

function column(Y: Matrix, index: number): Vector {
  const out = new Array<number>(Y.length);
  for (let i = 0; i < Y.length; i += 1) {
    out[i] = Y[i][index];
  }
  return out;
}

export class MultiTaskLasso {
  coef_: Matrix = [];
  intercept_: Vector = [];
  nIter_ = 0;

  private alpha: number;
  private fitIntercept: boolean;
  private maxIter: number;
  private tolerance: number;
  private estimators_: Lasso[] = [];
  private nFeaturesIn_: number | null = null;
  private nOutputs_: number | null = null;
  private fitted = false;

  constructor(options: MultiTaskLassoOptions = {}) {
    this.alpha = options.alpha ?? 1;
    this.fitIntercept = options.fitIntercept ?? true;
    this.maxIter = options.maxIter ?? 1000;
    this.tolerance = options.tolerance ?? 1e-4;
  }

  fit(X: Matrix, Y: Matrix): this {
    assertNonEmptyMatrix(Y, "Y");
    assertConsistentRowSize(Y, "Y");
    assertFiniteMatrix(Y, "Y");
    if (Y.length !== X.length) {
      throw new Error(`Y must have the same number of rows as X. Expected ${X.length}, got ${Y.length}.`);
    }
    validateRegressionInputs(X, column(Y, 0));

    this.nFeaturesIn_ = X[0].length;
    this.nOutputs_ = Y[0].length;
    this.estimators_ = new Array<Lasso>(this.nOutputs_);
    this.coef_ = new Array<Matrix[number]>(this.nOutputs_);
    this.intercept_ = new Array<number>(this.nOutputs_).fill(0);
    this.nIter_ = 0;

    for (let out = 0; out < this.nOutputs_; out += 1) {
      const estimator = new Lasso({
        alpha: this.alpha,
        fitIntercept: this.fitIntercept,
        maxIter: this.maxIter,
        tolerance: this.tolerance,
      }).fit(X, column(Y, out));
      this.estimators_[out] = estimator;
      this.coef_[out] = estimator.coef_.slice();
      this.intercept_[out] = estimator.intercept_;
      this.nIter_ = Math.max(this.nIter_, estimator.nIter_);
    }

    this.fitted = true;
    return this;
  }

  predict(X: Matrix): Matrix {
    this.assertFitted();
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }

    const out: Matrix = Array.from({ length: X.length }, () => new Array<number>(this.nOutputs_!).fill(0));
    for (let o = 0; o < this.nOutputs_!; o += 1) {
      const pred = this.estimators_[o].predict(X);
      for (let i = 0; i < pred.length; i += 1) {
        out[i][o] = pred[i];
      }
    }
    return out;
  }

  score(X: Matrix, Y: Matrix): number {
    return r2Score(Y, this.predict(X)) as number;
  }

  private assertFitted(): void {
    if (!this.fitted || this.nFeaturesIn_ === null || this.nOutputs_ === null) {
      throw new Error("MultiTaskLasso has not been fitted.");
    }
  }
}