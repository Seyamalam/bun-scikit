import type { Matrix, RegressionModel, Vector } from "../types";
import { r2Score } from "../metrics/regression";
import {
  addInterceptColumn,
  dot,
  inverseMatrix,
  mean,
  multiplyMatrices,
  multiplyMatrixVector,
  solveSymmetricPositiveDefinite,
  transpose,
} from "../utils/linalg";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  validateRegressionInputs,
} from "../utils/validation";

export interface LinearRegressionOptions {
  fitIntercept?: boolean;
  solver?: "normal" | "gd";
  learningRate?: number;
  maxIter?: number;
  tolerance?: number;
}

export class LinearRegression implements RegressionModel {
  coef_: Vector = [];
  intercept_ = 0;

  private readonly fitIntercept: boolean;
  private readonly solver: "normal" | "gd";
  private readonly learningRate: number;
  private readonly maxIter: number;
  private readonly tolerance: number;
  private isFitted = false;

  constructor(options: LinearRegressionOptions = {}) {
    this.fitIntercept = options.fitIntercept ?? true;
    this.solver = options.solver ?? "normal";
    this.learningRate = options.learningRate ?? 0.01;
    this.maxIter = options.maxIter ?? 10_000;
    this.tolerance = options.tolerance ?? 1e-8;
  }

  fit(X: Matrix, y: Vector): this {
    validateRegressionInputs(X, y);

    if (this.solver === "normal") {
      this.fitNormalEquation(X, y);
    } else {
      this.fitGradientDescent(X, y);
    }

    this.isFitted = true;
    return this;
  }

  predict(X: Matrix): Vector {
    if (!this.isFitted) {
      throw new Error("LinearRegression has not been fitted.");
    }

    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    if (X[0].length !== this.coef_.length) {
      throw new Error(
        `Feature size mismatch. Expected ${this.coef_.length}, got ${X[0].length}.`,
      );
    }

    return X.map((row) => this.intercept_ + dot(row, this.coef_));
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return r2Score(y, this.predict(X));
  }

  private fitNormalEquation(X: Matrix, y: Vector): void {
    const XDesign = this.fitIntercept ? addInterceptColumn(X) : X;
    const XT = transpose(XDesign);
    const XTX = multiplyMatrices(XT, XDesign);

    // Small diagonal regularization stabilizes inversion for near-singular matrices.
    const EPSILON = 1e-8;
    for (let i = 0; i < XTX.length; i += 1) {
      const isInterceptTerm = this.fitIntercept && i === 0;
      if (!isInterceptTerm) {
        XTX[i][i] += EPSILON;
      }
    }

    const XTy = multiplyMatrixVector(XT, y);
    let beta: Vector;
    try {
      beta = solveSymmetricPositiveDefinite(XTX, XTy);
    } catch {
      beta = multiplyMatrixVector(inverseMatrix(XTX), XTy);
    }

    if (this.fitIntercept) {
      this.intercept_ = beta[0];
      this.coef_ = beta.slice(1);
      return;
    }

    this.intercept_ = 0;
    this.coef_ = beta;
  }

  private fitGradientDescent(X: Matrix, y: Vector): void {
    const nSamples = X.length;
    const nFeatures = X[0].length;
    this.coef_ = new Array(nFeatures).fill(0);
    this.intercept_ = this.fitIntercept ? mean(y) : 0;

    let previousLoss = Number.POSITIVE_INFINITY;

    for (let iter = 0; iter < this.maxIter; iter += 1) {
      const predictions = X.map((row) => this.intercept_ + dot(row, this.coef_));
      const errors = predictions.map((pred, i) => pred - y[i]);

      const gradients = new Array(nFeatures).fill(0);
      let interceptGradient = 0;
      let loss = 0;

      for (let i = 0; i < nSamples; i += 1) {
        const error = errors[i];
        interceptGradient += error;
        loss += error * error;

        for (let j = 0; j < nFeatures; j += 1) {
          gradients[j] += error * X[i][j];
        }
      }

      loss /= nSamples;
      if (Math.abs(previousLoss - loss) < this.tolerance) {
        return;
      }
      previousLoss = loss;

      const scale = 2 / nSamples;
      for (let j = 0; j < nFeatures; j += 1) {
        this.coef_[j] -= this.learningRate * scale * gradients[j];
      }

      if (this.fitIntercept) {
        this.intercept_ -= this.learningRate * scale * interceptGradient;
      }
    }
  }
}
