import type { Matrix, RegressionModel, Vector } from "../types";
import { r2Score } from "../metrics/regression";
import { dot } from "../utils/linalg";
import { getZigKernels } from "../native/zigKernels";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  validateRegressionInputs,
} from "../utils/validation";

export interface LinearRegressionOptions {
  fitIntercept?: boolean;
  solver?: "normal";
}

export class LinearRegression implements RegressionModel {
  coef_: Vector = [];
  intercept_ = 0;
  fitBackend_: "zig" = "zig";
  fitBackendLibrary_: string | null = null;

  private readonly fitIntercept: boolean;
  private readonly solver: "normal";
  private isFitted = false;

  constructor(options: LinearRegressionOptions = {}) {
    this.fitIntercept = options.fitIntercept ?? true;
    this.solver = options.solver ?? "normal";
  }

  fit(X: Matrix, y: Vector): this {
    validateRegressionInputs(X, y);
    if (this.solver !== "normal") {
      throw new Error("LinearRegression solver 'normal' is required in zig-only mode.");
    }
    this.fitNormalEquationNative(X, y);

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
    if (!kernels) {
      throw new Error(
        "LinearRegression requires native Zig kernels. Build them with `bun run native:build`.",
      );
    }
    if (
      !kernels.linearModelCreate ||
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
