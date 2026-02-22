import type { Matrix, RegressionModel, Vector } from "../types";
import { r2Score } from "../metrics/regression";
import { dot, mean } from "../utils/linalg";
import { getZigKernels } from "../native/zigKernels";
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
  backend?: "auto" | "js" | "zig";
}

export class LinearRegression implements RegressionModel {
  coef_: Vector = [];
  intercept_ = 0;
  fitBackend_: "js" | "zig" = "js";
  fitBackendLibrary_: string | null = null;

  private readonly fitIntercept: boolean;
  private readonly solver: "normal" | "gd";
  private readonly learningRate: number;
  private readonly maxIter: number;
  private readonly tolerance: number;
  private readonly backend: "auto" | "js" | "zig";
  private isFitted = false;

  constructor(options: LinearRegressionOptions = {}) {
    this.fitIntercept = options.fitIntercept ?? true;
    this.solver = options.solver ?? "normal";
    this.learningRate = options.learningRate ?? 0.01;
    this.maxIter = options.maxIter ?? 10_000;
    this.tolerance = options.tolerance ?? 1e-8;
    this.backend = options.backend ?? "auto";
  }

  fit(X: Matrix, y: Vector): this {
    validateRegressionInputs(X, y);

    if (this.solver === "normal") {
      const kernels = this.backend !== "js" ? getZigKernels() : null;
      if (this.backend === "zig" && !kernels) {
        throw new Error(
          "LinearRegression backend 'zig' requested but native kernels were not found. Build them with `bun run native:build`.",
        );
      }

      if (
        kernels?.linearModelCreate &&
        kernels.linearModelDestroy &&
        kernels.linearModelFit &&
        kernels.linearModelCopyCoefficients &&
        kernels.linearModelGetIntercept
      ) {
        try {
          this.fitNormalEquationNative(X, y);
        } catch (error) {
          if (this.backend === "zig") {
            throw error;
          }
          this.fitNormalEquation(X, y);
        }
      } else {
        this.fitNormalEquation(X, y);
      }
    } else {
      this.fitGradientDescent(X, y);
      this.fitBackend_ = "js";
      this.fitBackendLibrary_ = null;
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

  private fitNormalEquationNative(X: Matrix, y: Vector): void {
    const kernels = getZigKernels();
    if (
      !kernels?.linearModelCreate ||
      !kernels.linearModelDestroy ||
      !kernels.linearModelFit ||
      !kernels.linearModelCopyCoefficients ||
      !kernels.linearModelGetIntercept
    ) {
      throw new Error("Native linear model symbols are not available.");
    }

    const nSamples = X.length;
    const nFeatures = X[0].length;
    const flattenedX = this.flattenMatrix(X);
    const yBuffer = this.toFloat64Vector(y);
    const handle = kernels.linearModelCreate(nFeatures, this.fitIntercept ? 1 : 0);
    if (handle === 0n) {
      throw new Error("Failed to create native linear model handle.");
    }

    try {
      const fitStatus = kernels.linearModelFit(handle, flattenedX, yBuffer, nSamples, 1e-8);
      if (fitStatus !== 1) {
        throw new Error("Native linear model fit failed.");
      }

      const coefficients = new Float64Array(nFeatures);
      const copied = kernels.linearModelCopyCoefficients(handle, coefficients);
      if (copied !== 1) {
        throw new Error("Failed to copy native linear coefficients.");
      }

      this.coef_ = Array.from(coefficients);
      this.intercept_ = kernels.linearModelGetIntercept(handle);
      this.fitBackend_ = "zig";
      this.fitBackendLibrary_ = kernels.libraryPath;
    } catch (error) {
      kernels.linearModelDestroy(handle);
      throw error;
    }
    kernels.linearModelDestroy(handle);
  }

  private fitNormalEquation(X: Matrix, y: Vector): void {
    const sampleCount = X.length;
    const featureCount = X[0].length;
    const dim = this.fitIntercept ? featureCount + 1 : featureCount;
    const gram = new Float64Array(dim * dim);
    const rhs = new Float64Array(dim);
    const fitIntercept = this.fitIntercept;
    const offset = fitIntercept ? 1 : 0;

    for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex += 1) {
      const row = X[sampleIndex];
      const target = y[sampleIndex];

      for (let i = 0; i < dim; i += 1) {
        const xi = fitIntercept && i === 0 ? 1 : row[i - offset];
        rhs[i] += xi * target;

        const rowOffset = i * dim;
        for (let j = 0; j <= i; j += 1) {
          const xj = fitIntercept && j === 0 ? 1 : row[j - offset];
          gram[rowOffset + j] += xi * xj;
        }
      }
    }

    for (let i = 0; i < dim; i += 1) {
      for (let j = 0; j < i; j += 1) {
        gram[j * dim + i] = gram[i * dim + j];
      }
    }

    const baseGram = gram.slice();
    const baseRegularization = 1e-8;
    const maxAttempts = 4;
    let beta: Float64Array | null = null;

    for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
      const gramAttempt = baseGram.slice();
      const regularization = baseRegularization * 10 ** attempt;
      for (let i = 0; i < dim; i += 1) {
        const isInterceptTerm = fitIntercept && i === 0;
        if (!isInterceptTerm) {
          gramAttempt[i * dim + i] += regularization;
        }
      }

      beta = this.solveSymmetricPositiveDefiniteDense(gramAttempt, rhs, dim);
      if (beta) {
        break;
      }
    }

    if (!beta) {
      throw new Error(
        "LinearRegression normal solver failed: matrix is not positive definite after regularization.",
      );
    }

    if (fitIntercept) {
      this.intercept_ = beta[0];
      this.coef_ = Array.from(beta.subarray(1));
    } else {
      this.intercept_ = 0;
      this.coef_ = Array.from(beta);
    }

    this.fitBackend_ = "js";
    this.fitBackendLibrary_ = null;
  }

  private solveSymmetricPositiveDefiniteDense(
    gram: Float64Array,
    rhs: Float64Array,
    dim: number,
  ): Float64Array | null {
    const lower = new Float64Array(dim * dim);
    const EPSILON = 1e-12;

    for (let i = 0; i < dim; i += 1) {
      const rowOffsetI = i * dim;
      for (let j = 0; j <= i; j += 1) {
        const rowOffsetJ = j * dim;
        let sum = gram[rowOffsetI + j];
        for (let k = 0; k < j; k += 1) {
          sum -= lower[rowOffsetI + k] * lower[rowOffsetJ + k];
        }

        if (i === j) {
          if (sum <= EPSILON) {
            return null;
          }
          lower[rowOffsetI + j] = Math.sqrt(sum);
        } else {
          lower[rowOffsetI + j] = sum / lower[rowOffsetJ + j];
        }
      }
    }

    const forward = new Float64Array(dim);
    for (let i = 0; i < dim; i += 1) {
      const rowOffset = i * dim;
      let sum = rhs[i];
      for (let k = 0; k < i; k += 1) {
        sum -= lower[rowOffset + k] * forward[k];
      }
      forward[i] = sum / lower[rowOffset + i];
    }

    const solution = new Float64Array(dim);
    for (let i = dim - 1; i >= 0; i -= 1) {
      let sum = forward[i];
      for (let k = i + 1; k < dim; k += 1) {
        sum -= lower[k * dim + i] * solution[k];
      }
      solution[i] = sum / lower[i * dim + i];
    }

    return solution;
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

  private flattenMatrix(X: Matrix): Float64Array {
    const rowCount = X.length;
    const featureCount = X[0].length;
    const flattenedX = new Float64Array(rowCount * featureCount);
    for (let i = 0; i < rowCount; i += 1) {
      const row = X[i];
      const rowOffset = i * featureCount;
      for (let j = 0; j < featureCount; j += 1) {
        flattenedX[rowOffset + j] = row[j];
      }
    }
    return flattenedX;
  }

  private toFloat64Vector(y: Vector): Float64Array {
    const yBuffer = new Float64Array(y.length);
    for (let i = 0; i < y.length; i += 1) {
      yBuffer[i] = y[i];
    }
    return yBuffer;
  }

}
