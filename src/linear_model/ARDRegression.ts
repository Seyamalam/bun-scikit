import type { Matrix, RegressionModel, Vector } from "../types";
import { r2Score } from "../metrics/regression";
import { dot, identityMatrix } from "../utils/linalg";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  validateRegressionInputs,
} from "../utils/validation";
import { Ridge } from "./Ridge";

export interface ARDRegressionOptions {
  alpha1?: number;
  alpha2?: number;
  lambda1?: number;
  lambda2?: number;
  thresholdLambda?: number;
  nIter?: number;
  tolerance?: number;
  fitIntercept?: boolean;
  computeScore?: boolean;
}

export class ARDRegression implements RegressionModel {
  coef_: Vector = [];
  intercept_ = 0;
  alpha_ = 1;
  lambda_: Vector = [];
  sigma_: Matrix | null = null;
  scores_: Vector = [];
  nIter_ = 0;

  private alpha1: number;
  private alpha2: number;
  private lambda1: number;
  private lambda2: number;
  private thresholdLambda: number;
  private nIter: number;
  private tolerance: number;
  private fitIntercept: boolean;
  private computeScore: boolean;
  private fitted = false;

  constructor(options: ARDRegressionOptions = {}) {
    this.alpha1 = options.alpha1 ?? 1e-6;
    this.alpha2 = options.alpha2 ?? 1e-6;
    this.lambda1 = options.lambda1 ?? 1e-6;
    this.lambda2 = options.lambda2 ?? 1e-6;
    this.thresholdLambda = options.thresholdLambda ?? 10_000;
    this.nIter = options.nIter ?? 300;
    this.tolerance = options.tolerance ?? 1e-4;
    this.fitIntercept = options.fitIntercept ?? true;
    this.computeScore = options.computeScore ?? false;
    this.validateOptions();
  }

  fit(X: Matrix, y: Vector, sampleWeight?: Vector): this {
    validateRegressionInputs(X, y);

    const ridge = new Ridge({
      alpha: Math.max(1e-9, this.lambda1 / Math.max(this.alpha1, 1e-9)),
      fitIntercept: this.fitIntercept,
    }).fit(X, y, sampleWeight);

    this.coef_ = ridge.coef_.slice();
    this.intercept_ = ridge.intercept_;

    let residual = 0;
    for (let i = 0; i < X.length; i += 1) {
      const diff = y[i] - (this.intercept_ + dot(X[i], this.coef_));
      residual += diff * diff;
    }

    this.alpha_ = (X.length + 2 * this.alpha1) / Math.max(residual + 2 * this.alpha2, 1e-9);
    this.lambda_ = new Array<number>(this.coef_.length);
    for (let i = 0; i < this.coef_.length; i += 1) {
      const precision = (1 + 2 * this.lambda1) / Math.max(this.coef_[i] * this.coef_[i] + 2 * this.lambda2, 1e-9);
      this.lambda_[i] = Math.min(precision, this.thresholdLambda);
    }

    this.sigma_ = identityMatrix(this.coef_.length);
    for (let i = 0; i < this.coef_.length; i += 1) {
      this.sigma_[i][i] = 1 / Math.max(this.lambda_[i], 1e-9);
    }

    this.scores_ = this.computeScore ? [-0.5 * residual] : [];
    this.nIter_ = 1;
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

  private assertFitted(): void {
    if (!this.fitted || this.sigma_ === null) {
      throw new Error("ARDRegression has not been fitted.");
    }
  }

  private validateOptions(): void {
    const values = [
      this.alpha1,
      this.alpha2,
      this.lambda1,
      this.lambda2,
      this.thresholdLambda,
      this.tolerance,
    ];
    for (let i = 0; i < values.length; i += 1) {
      if (!Number.isFinite(values[i]) || values[i] < 0) {
        throw new Error("ARDRegression options must be finite and non-negative.");
      }
    }
    if (!Number.isInteger(this.nIter) || this.nIter < 1) {
      throw new Error(`nIter must be an integer >= 1. Got ${this.nIter}.`);
    }
  }
}