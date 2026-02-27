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
import {
  argmax,
  normalizeProbabilitiesInPlace,
  uniqueSortedLabels,
} from "../utils/classification";

export interface LogisticRegressionOptions {
  fitIntercept?: boolean;
  solver?: "gd" | "lbfgs";
  learningRate?: number;
  maxIter?: number;
  tolerance?: number;
  l2?: number;
  lbfgsMemory?: number;
}

function sigmoid(z: number): number {
  if (z >= 0) {
    const expNeg = Math.exp(-z);
    return 1 / (1 + expNeg);
  }
  const expPos = Math.exp(z);
  return expPos / (1 + expPos);
}

interface BinaryFitResult {
  coef: Vector;
  intercept: number;
  fitBackend: "zig";
  fitBackendLibrary: string | null;
}

export class LogisticRegression implements ClassificationModel {
  coef_: Vector | Matrix = [];
  intercept_: number | Vector = 0;
  classes_: Vector = [0, 1];
  fitBackend_: "zig" = "zig";
  fitBackendLibrary_: string | null = null;

  private readonly fitIntercept: boolean;
  private readonly solver: "gd" | "lbfgs";
  private readonly learningRate: number;
  private readonly maxIter: number;
  private readonly tolerance: number;
  private readonly l2: number;
  private readonly lbfgsMemory: number;
  private isFitted = false;
  private featureCount = 0;
  private coefMatrix_: Matrix = [];
  private interceptVector_: Vector = [];

  constructor(options: LogisticRegressionOptions = {}) {
    this.fitIntercept = options.fitIntercept ?? true;
    this.solver = options.solver ?? "gd";
    this.learningRate = options.learningRate ?? 0.1;
    this.maxIter = options.maxIter ?? 20_000;
    this.tolerance = options.tolerance ?? 1e-8;
    this.l2 = options.l2 ?? 0;
    this.lbfgsMemory = options.lbfgsMemory ?? 7;
  }

  fit(X: Matrix, y: Vector, sampleWeight?: Vector): this {
    validateClassificationInputs(X, y);

    this.classes_ = uniqueSortedLabels(y);
    if (this.classes_.length < 2) {
      throw new Error("LogisticRegression requires at least two classes.");
    }

    const nSamples = X.length;
    const nFeatures = X[0].length;
    this.featureCount = nFeatures;
    const flattenedX = this.flattenMatrix(X);

    this.coefMatrix_ = new Array<Matrix[number]>(this.classes_.length);
    this.interceptVector_ = new Array<number>(this.classes_.length).fill(0);

    for (let classIndex = 0; classIndex < this.classes_.length; classIndex += 1) {
      const positiveLabel = this.classes_[classIndex];
      const binaryTargets = new Float64Array(nSamples);
      for (let i = 0; i < nSamples; i += 1) {
        binaryTargets[i] = y[i] === positiveLabel ? 1 : 0;
      }
      const fitResult = this.fitBinary(flattenedX, binaryTargets, nSamples, nFeatures);
      this.coefMatrix_[classIndex] = fitResult.coef;
      this.interceptVector_[classIndex] = fitResult.intercept;
      this.fitBackend_ = fitResult.fitBackend;
      this.fitBackendLibrary_ = fitResult.fitBackendLibrary;
    }

    if (this.classes_.length === 2) {
      this.coef_ = this.coefMatrix_[1].slice();
      this.intercept_ = this.interceptVector_[1];
    } else {
      this.coef_ = this.coefMatrix_.map((row) => row.slice());
      this.intercept_ = this.interceptVector_.slice();
    }

    this.isFitted = true;
    return this;
  }

  predictProba(X: Matrix): Matrix {
    this.assertFitted();
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    if (X[0].length !== this.featureCount) {
      throw new Error(
        `Feature size mismatch. Expected ${this.featureCount}, got ${X[0].length}.`,
      );
    }

    const out: Matrix = new Array(X.length);
    for (let i = 0; i < X.length; i += 1) {
      const rowScores = new Array<number>(this.classes_.length);
      for (let classIndex = 0; classIndex < this.classes_.length; classIndex += 1) {
        rowScores[classIndex] = sigmoid(
          this.interceptVector_[classIndex] + dot(X[i], this.coefMatrix_[classIndex]),
        );
      }
      normalizeProbabilitiesInPlace(rowScores);
      out[i] = rowScores;
    }
    return out;
  }

  predict(X: Matrix): Vector {
    return this.predictProba(X).map((row) => this.classes_[argmax(row)]);
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return accuracyScore(y, this.predict(X));
  }

  private fitBinary(
    flattenedX: Float64Array,
    yBuffer: Float64Array,
    nSamples: number,
    nFeatures: number,
  ): BinaryFitResult {
    const coefficients = new Float64Array(nFeatures);
    const gradients = new Float64Array(nFeatures);
    const intercept = new Float64Array(1);

    const kernels = getZigKernels();
    if (!kernels) {
      throw new Error(
        "LogisticRegression requires native Zig kernels. Build them with `bun run native:build`.",
      );
    }

    if (
      kernels.logisticModelCreate &&
      kernels.logisticModelDestroy &&
      kernels.logisticModelCopyCoefficients &&
      kernels.logisticModelGetIntercept
    ) {
      const fitNative =
        this.solver === "lbfgs" ? kernels.logisticModelFitLbfgs : kernels.logisticModelFit;
      if (!fitNative) {
        throw new Error(
          `LogisticRegression solver '${this.solver}' is unavailable in native kernels.`,
        );
      }

      const handle = kernels.logisticModelCreate(nFeatures, this.fitIntercept ? 1 : 0);
      if (handle === 0n) {
        throw new Error("Failed to create native logistic model handle.");
      }

      try {
        const epochsRan =
          this.solver === "lbfgs"
            ? kernels.logisticModelFitLbfgs!(
                handle,
                flattenedX,
                yBuffer,
                nSamples,
                this.maxIter,
                this.tolerance,
                this.l2,
                this.lbfgsMemory,
              )
            : kernels.logisticModelFit!(
                handle,
                flattenedX,
                yBuffer,
                nSamples,
                this.learningRate,
                this.l2,
                this.maxIter,
                this.tolerance,
              );
        if (epochsRan === 0n && this.maxIter > 0) {
          throw new Error("Native logistic model fit failed.");
        }

        const copied = kernels.logisticModelCopyCoefficients(handle, coefficients);
        if (copied !== 1) {
          throw new Error("Failed to copy native logistic coefficients.");
        }

        const result: BinaryFitResult = {
          coef: Array.from(coefficients),
          intercept: kernels.logisticModelGetIntercept(handle),
          fitBackend: "zig",
          fitBackendLibrary: kernels.libraryPath,
        };
        kernels.logisticModelDestroy(handle);
        return result;
      } catch (error) {
        kernels.logisticModelDestroy(handle);
        throw error;
      }
    }

    if (this.solver === "lbfgs") {
      throw new Error("LogisticRegression solver 'lbfgs' requires native model-handle kernels.");
    }

    if (kernels.logisticTrainEpoch) {
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

      return {
        coef: Array.from(coefficients),
        intercept: intercept[0],
        fitBackend: "zig",
        fitBackendLibrary: kernels.libraryPath,
      };
    }

    throw new Error(
      "Native logistic kernels are unavailable. Rebuild with `bun run native:build` and ensure model-handle or epoch kernels are exported.",
    );
  }

  private assertFitted(): void {
    if (!this.isFitted || this.coefMatrix_.length === 0) {
      throw new Error("LogisticRegression has not been fitted.");
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
}

