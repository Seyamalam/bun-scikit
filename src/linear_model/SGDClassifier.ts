import type { ClassificationModel, Matrix, Vector } from "../types";
import { accuracyScore } from "../metrics/classification";
import { dot } from "../utils/linalg";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  validateClassificationInputs,
} from "../utils/validation";
import {
  argmax,
  normalizeProbabilitiesInPlace,
  uniqueSortedLabels,
} from "../utils/classification";

export type SGDClassifierLoss = "hinge" | "log_loss";

export interface SGDClassifierOptions {
  loss?: SGDClassifierLoss;
  fitIntercept?: boolean;
  learningRate?: number;
  maxIter?: number;
  tolerance?: number;
  l2?: number;
}

function sigmoid(z: number): number {
  if (z >= 0) {
    const expNeg = Math.exp(-z);
    return 1 / (1 + expNeg);
  }
  const expPos = Math.exp(z);
  return expPos / (1 + expPos);
}

interface BinarySgdResult {
  coef: Vector;
  intercept: number;
}

export class SGDClassifier implements ClassificationModel {
  coef_: Vector | Matrix = [];
  intercept_: number | Vector = 0;
  classes_: Vector = [0, 1];

  private readonly loss: SGDClassifierLoss;
  private readonly fitIntercept: boolean;
  private readonly learningRate: number;
  private readonly maxIter: number;
  private readonly tolerance: number;
  private readonly l2: number;
  private isFitted = false;
  private featureCount = 0;
  private coefMatrix_: Matrix = [];
  private interceptVector_: Vector = [];

  constructor(options: SGDClassifierOptions = {}) {
    this.loss = options.loss ?? "hinge";
    this.fitIntercept = options.fitIntercept ?? true;
    this.learningRate = options.learningRate ?? 0.05;
    this.maxIter = options.maxIter ?? 10_000;
    this.tolerance = options.tolerance ?? 1e-6;
    this.l2 = options.l2 ?? 0;
  }

  fit(X: Matrix, y: Vector, sampleWeight?: Vector): this {
    validateClassificationInputs(X, y);
    this.classes_ = uniqueSortedLabels(y);
    if (this.classes_.length < 2) {
      throw new Error("SGDClassifier requires at least two classes.");
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

  predictProba(X: Matrix): Matrix {
    if (this.loss !== "log_loss") {
      throw new Error("predictProba is only available when loss='log_loss'.");
    }
    this.assertFitted();
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.featureCount) {
      throw new Error(`Feature size mismatch. Expected ${this.featureCount}, got ${X[0].length}.`);
    }

    const out: Matrix = new Array(X.length);
    for (let i = 0; i < X.length; i += 1) {
      const row = new Array<number>(this.classes_.length);
      for (let classIndex = 0; classIndex < this.classes_.length; classIndex += 1) {
        row[classIndex] = sigmoid(dot(X[i], this.coefMatrix_[classIndex]) + this.interceptVector_[classIndex]);
      }
      normalizeProbabilitiesInPlace(row);
      out[i] = row;
    }
    return out;
  }

  predict(X: Matrix): Vector {
    this.assertFitted();
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.featureCount) {
      throw new Error(`Feature size mismatch. Expected ${this.featureCount}, got ${X[0].length}.`);
    }

    if (this.loss === "log_loss") {
      return this.predictProba(X).map((row) => this.classes_[argmax(row)]);
    }

    return X.map((row) => {
      const scores = new Array<number>(this.classes_.length);
      for (let classIndex = 0; classIndex < this.classes_.length; classIndex += 1) {
        scores[classIndex] = dot(row, this.coefMatrix_[classIndex]) + this.interceptVector_[classIndex];
      }
      return this.classes_[argmax(scores)];
    });
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return accuracyScore(y, this.predict(X));
  }

  private assertFitted(): void {
    if (!this.isFitted || this.coefMatrix_.length === 0) {
      throw new Error("SGDClassifier has not been fitted.");
    }
  }

  private fitBinary(X: Matrix, y: Vector): BinarySgdResult {
    const nSamples = X.length;
    const nFeatures = X[0].length;
    const ySigned = y.map((value) => (value === 1 ? 1 : -1));

    const coef = new Array<number>(nFeatures).fill(0);
    let intercept = 0;

    for (let iter = 0; iter < this.maxIter; iter += 1) {
      const gradients = new Array<number>(nFeatures).fill(0);
      let interceptGradient = 0;

      for (let i = 0; i < nSamples; i += 1) {
        const score = dot(X[i], coef) + intercept;

        if (this.loss === "hinge") {
          const margin = ySigned[i] * score;
          if (margin < 1) {
            const factor = -ySigned[i];
            for (let j = 0; j < nFeatures; j += 1) {
              gradients[j] += factor * X[i][j];
            }
            if (this.fitIntercept) {
              interceptGradient += factor;
            }
          }
        } else {
          const p = sigmoid(score);
          const error = p - y[i];
          for (let j = 0; j < nFeatures; j += 1) {
            gradients[j] += error * X[i][j];
          }
          if (this.fitIntercept) {
            interceptGradient += error;
          }
        }
      }

      let maxUpdate = 0;
      for (let j = 0; j < nFeatures; j += 1) {
        const grad = gradients[j] / nSamples + this.l2 * coef[j];
        const delta = this.learningRate * grad;
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

