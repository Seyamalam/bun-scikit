import type { ClassificationModel, Matrix, Vector } from "../types";
import { dot } from "../utils/linalg";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  validateClassificationInputs,
} from "../utils/validation";
import { accuracyScore } from "../metrics/classification";
import { getZigKernels } from "../native/zigKernels";

export interface LogisticRegressionOptions {
  fitIntercept?: boolean;
  learningRate?: number;
  maxIter?: number;
  tolerance?: number;
  l2?: number;
  backend?: "auto" | "js" | "zig";
}

function sigmoid(z: number): number {
  if (z >= 0) {
    const expNeg = Math.exp(-z);
    return 1 / (1 + expNeg);
  }
  const expPos = Math.exp(z);
  return expPos / (1 + expPos);
}

export class LogisticRegression implements ClassificationModel {
  coef_: Vector = [];
  intercept_ = 0;
  classes_: Vector = [0, 1];
  fitBackend_: "js" | "zig" = "js";
  fitBackendLibrary_: string | null = null;

  private readonly fitIntercept: boolean;
  private readonly learningRate: number;
  private readonly maxIter: number;
  private readonly tolerance: number;
  private readonly l2: number;
  private readonly backend: "auto" | "js" | "zig";
  private isFitted = false;

  constructor(options: LogisticRegressionOptions = {}) {
    this.fitIntercept = options.fitIntercept ?? true;
    this.learningRate = options.learningRate ?? 0.1;
    this.maxIter = options.maxIter ?? 20_000;
    this.tolerance = options.tolerance ?? 1e-8;
    this.l2 = options.l2 ?? 0;
    this.backend = options.backend ?? "auto";
  }

  fit(X: Matrix, y: Vector): this {
    validateClassificationInputs(X, y);

    const nSamples = X.length;
    const nFeatures = X[0].length;
    const flattenedX = new Float64Array(nSamples * nFeatures);
    for (let i = 0; i < nSamples; i += 1) {
      const row = X[i];
      const rowOffset = i * nFeatures;
      for (let j = 0; j < nFeatures; j += 1) {
        flattenedX[rowOffset + j] = row[j];
      }
    }

    const yBuffer = new Float64Array(nSamples);
    for (let i = 0; i < nSamples; i += 1) {
      yBuffer[i] = y[i];
    }

    const coefficients = new Float64Array(nFeatures);
    const gradients = new Float64Array(nFeatures);
    const intercept = new Float64Array(1);

    const kernels = this.backend !== "js" ? getZigKernels() : null;
    if (this.backend === "zig" && !kernels) {
      throw new Error(
        "LogisticRegression backend 'zig' requested but native kernels were not found. Build them with `bun run native:build`.",
      );
    }

    if (kernels) {
      if (kernels.logisticTrainEpochs) {
        kernels.logisticTrainEpochs(
          flattenedX,
          yBuffer,
          nSamples,
          nFeatures,
          coefficients,
          intercept,
          gradients,
          this.learningRate,
          this.l2,
          this.fitIntercept ? 1 : 0,
          this.maxIter,
          this.tolerance,
        );
      } else {
        for (let iter = 0; iter < this.maxIter; iter += 1) {
          const maxUpdate = kernels.logisticTrainEpoch(
            flattenedX,
            yBuffer,
            nSamples,
            nFeatures,
            coefficients,
            intercept,
            gradients,
            this.learningRate,
            this.l2,
            this.fitIntercept ? 1 : 0,
          );

          if (maxUpdate < this.tolerance) {
            break;
          }
        }
      }

      this.fitBackend_ = "zig";
      this.fitBackendLibrary_ = kernels.libraryPath;
      this.coef_ = Array.from(coefficients);
      this.intercept_ = intercept[0];
      this.isFitted = true;
      return this;
    }

    let interceptScalar = 0;
    const scale = 1 / nSamples;

    for (let iter = 0; iter < this.maxIter; iter += 1) {
      let interceptGradient = 0;
      gradients.fill(0);

      for (let i = 0; i < nSamples; i += 1) {
        const rowOffset = i * nFeatures;
        let z = interceptScalar;
        for (let j = 0; j < nFeatures; j += 1) {
          z += flattenedX[rowOffset + j] * coefficients[j];
        }
        const prediction = sigmoid(z);
        const target = yBuffer[i];
        const error = prediction - target;
        interceptGradient += error;

        for (let j = 0; j < nFeatures; j += 1) {
          gradients[j] += error * flattenedX[rowOffset + j];
        }
      }

      let maxUpdate = 0;
      for (let j = 0; j < nFeatures; j += 1) {
        const l2Term = this.l2 > 0 ? this.l2 * coefficients[j] : 0;
        const delta = this.learningRate * (scale * gradients[j] + scale * l2Term);
        coefficients[j] -= delta;
        const absDelta = Math.abs(delta);
        if (absDelta > maxUpdate) {
          maxUpdate = absDelta;
        }
      }

      if (this.fitIntercept) {
        const interceptDelta = this.learningRate * scale * interceptGradient;
        interceptScalar -= interceptDelta;
        const absInterceptDelta = Math.abs(interceptDelta);
        if (absInterceptDelta > maxUpdate) {
          maxUpdate = absInterceptDelta;
        }
      }

      if (maxUpdate < this.tolerance) {
        break;
      }
    }

    this.fitBackend_ = "js";
    this.fitBackendLibrary_ = null;
    this.coef_ = Array.from(coefficients);
    this.intercept_ = interceptScalar;
    this.isFitted = true;
    return this;
  }

  predictProba(X: Matrix): Matrix {
    if (!this.isFitted) {
      throw new Error("LogisticRegression has not been fitted.");
    }

    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    if (X[0].length !== this.coef_.length) {
      throw new Error(
        `Feature size mismatch. Expected ${this.coef_.length}, got ${X[0].length}.`,
      );
    }

    return X.map((row) => {
      const positive = sigmoid(this.intercept_ + dot(row, this.coef_));
      return [1 - positive, positive];
    });
  }

  predict(X: Matrix): Vector {
    return this.predictProba(X).map((pair) => (pair[1] >= 0.5 ? 1 : 0));
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return accuracyScore(y, this.predict(X));
  }
}
