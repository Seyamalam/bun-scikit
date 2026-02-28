import type { Matrix, RegressionModel, Vector } from "../types";
import { r2Score } from "../metrics/regression";
import { dot } from "../utils/linalg";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  validateRegressionInputs,
} from "../utils/validation";
import { fitRidgeClosedForm } from "./sharedRegularized";

export interface RidgeOptions {
  alpha?: number;
  fitIntercept?: boolean;
}

export class Ridge implements RegressionModel {
  coef_: Vector = [];
  intercept_ = 0;

  private alpha: number;
  private fitIntercept: boolean;
  private fitted = false;

  constructor(options: RidgeOptions = {}) {
    this.alpha = options.alpha ?? 1;
    this.fitIntercept = options.fitIntercept ?? true;
    this.validateOptions();
  }

  fit(X: Matrix, y: Vector, sampleWeight?: Vector): this {
    validateRegressionInputs(X, y);
    const model = fitRidgeClosedForm(X, y, this.alpha, this.fitIntercept);
    this.coef_ = model.coef;
    this.intercept_ = model.intercept;
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
  }

  private assertFitted(): void {
    if (!this.fitted) {
      throw new Error("Ridge has not been fitted.");
    }
  }
}
