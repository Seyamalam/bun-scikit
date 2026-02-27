import type { ClassificationModel, Matrix, Vector } from "../types";
import { accuracyScore } from "../metrics/classification";
import { dot } from "../utils/linalg";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  validateClassificationInputs,
} from "../utils/validation";
import { argmax, uniqueSortedLabels } from "../utils/classification";

export interface LinearSVCOptions {
  fitIntercept?: boolean;
  C?: number;
  learningRate?: number;
  maxIter?: number;
  tolerance?: number;
}

interface BinarySvcResult {
  coef: Vector;
  intercept: number;
}

export class LinearSVC implements ClassificationModel {
  coef_: Vector | Matrix = [];
  intercept_: number | Vector = 0;
  classes_: Vector = [0, 1];

  private readonly fitIntercept: boolean;
  private readonly C: number;
  private readonly learningRate: number;
  private readonly maxIter: number;
  private readonly tolerance: number;
  private isFitted = false;
  private featureCount = 0;
  private coefMatrix_: Matrix = [];
  private interceptVector_: Vector = [];

  constructor(options: LinearSVCOptions = {}) {
    this.fitIntercept = options.fitIntercept ?? true;
    this.C = options.C ?? 1.0;
    this.learningRate = options.learningRate ?? 0.05;
    this.maxIter = options.maxIter ?? 10_000;
    this.tolerance = options.tolerance ?? 1e-6;

    if (!Number.isFinite(this.C) || this.C <= 0) {
      throw new Error(`C must be > 0. Got ${this.C}.`);
    }
  }

  fit(X: Matrix, y: Vector, sampleWeight?: Vector): this {
    validateClassificationInputs(X, y);
    this.classes_ = uniqueSortedLabels(y);
    if (this.classes_.length < 2) {
      throw new Error("LinearSVC requires at least two classes.");
    }
    this.featureCount = X[0].length;

    this.coefMatrix_ = new Array<Matrix[number]>(this.classes_.length);
    this.interceptVector_ = new Array<number>(this.classes_.length).fill(0);
    for (let classIndex = 0; classIndex < this.classes_.length; classIndex += 1) {
      const label = this.classes_[classIndex];
      const binaryY = y.map((value) => (value === label ? 1 : 0));
      const result = this.fitBinary(X, binaryY);
      this.coefMatrix_[classIndex] = result.coef;
      this.interceptVector_[classIndex] = result.intercept;
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

  decisionFunction(X: Matrix): Vector | Matrix {
    this.assertFitted();
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.featureCount) {
      throw new Error(`Feature size mismatch. Expected ${this.featureCount}, got ${X[0].length}.`);
    }

    if (this.classes_.length === 2) {
      return X.map((row) => dot(row, this.coefMatrix_[1]) + this.interceptVector_[1]);
    }

    const out: Matrix = new Array(X.length);
    for (let i = 0; i < X.length; i += 1) {
      const row = new Array<number>(this.classes_.length);
      for (let classIndex = 0; classIndex < this.classes_.length; classIndex += 1) {
        row[classIndex] = dot(X[i], this.coefMatrix_[classIndex]) + this.interceptVector_[classIndex];
      }
      out[i] = row;
    }
    return out;
  }

  predict(X: Matrix): Vector {
    const decision = this.decisionFunction(X);
    if (Array.isArray(decision[0])) {
      const matrix = decision as Matrix;
      return matrix.map((row) => this.classes_[argmax(row)]);
    }
    return (decision as Vector).map((score) => (score >= 0 ? this.classes_[1] : this.classes_[0]));
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return accuracyScore(y, this.predict(X));
  }

  private assertFitted(): void {
    if (!this.isFitted || this.coefMatrix_.length === 0) {
      throw new Error("LinearSVC has not been fitted.");
    }
  }

  private fitBinary(X: Matrix, y: Vector): BinarySvcResult {
    const nSamples = X.length;
    const nFeatures = X[0].length;
    const coef = new Array<number>(nFeatures).fill(0);
    let intercept = 0;
    const ySigned = y.map((value) => (value === 1 ? 1 : -1));

    for (let iter = 0; iter < this.maxIter; iter += 1) {
      const gradients = coef.slice();
      let interceptGradient = 0;

      for (let i = 0; i < nSamples; i += 1) {
        const margin = ySigned[i] * (dot(X[i], coef) + intercept);
        if (margin < 1) {
          const factor = -this.C * ySigned[i];
          for (let j = 0; j < nFeatures; j += 1) {
            gradients[j] += factor * X[i][j];
          }
          if (this.fitIntercept) {
            interceptGradient += factor;
          }
        }
      }

      let maxUpdate = 0;
      for (let j = 0; j < nFeatures; j += 1) {
        const delta = this.learningRate * (gradients[j] / nSamples);
        coef[j] -= delta;
        const absDelta = Math.abs(delta);
        if (absDelta > maxUpdate) {
          maxUpdate = absDelta;
        }
      }

      if (this.fitIntercept) {
        const interceptDelta = this.learningRate * (interceptGradient / nSamples);
        intercept -= interceptDelta;
        const absInterceptDelta = Math.abs(interceptDelta);
        if (absInterceptDelta > maxUpdate) {
          maxUpdate = absInterceptDelta;
        }
      }

      if (maxUpdate < this.tolerance) {
        break;
      }
    }

    return { coef, intercept };
  }
}

