import type { ClassificationModel, Matrix, Vector } from "../types";
import { accuracyScore } from "../metrics/classification";
import { argmax, uniqueSortedLabels } from "../utils/classification";
import { dot } from "../utils/linalg";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  validateClassificationInputs,
} from "../utils/validation";

export type PassiveAggressiveLoss = "hinge" | "squared_hinge";

export interface PassiveAggressiveClassifierOptions {
  C?: number;
  fitIntercept?: boolean;
  maxIter?: number;
  tolerance?: number;
  loss?: PassiveAggressiveLoss;
}

export class PassiveAggressiveClassifier implements ClassificationModel {
  coef_: Vector | Matrix = [];
  intercept_: number | Vector = 0;
  classes_: Vector = [];
  nIter_ = 0;

  private C: number;
  private fitIntercept: boolean;
  private maxIter: number;
  private tolerance: number;
  private loss: PassiveAggressiveLoss;
  private coefMatrix_: Matrix = [];
  private interceptVector_: Vector = [];
  private fitted = false;
  private featureCount = 0;

  constructor(options: PassiveAggressiveClassifierOptions = {}) {
    this.C = options.C ?? 1;
    this.fitIntercept = options.fitIntercept ?? true;
    this.maxIter = options.maxIter ?? 1000;
    this.tolerance = options.tolerance ?? 1e-4;
    this.loss = options.loss ?? "hinge";
    this.validateOptions();
  }

  fit(X: Matrix, y: Vector, sampleWeight?: Vector): this {
    validateClassificationInputs(X, y);
    this.classes_ = uniqueSortedLabels(y);
    this.featureCount = X[0].length;

    const nClasses = this.classes_.length;
    this.coefMatrix_ = Array.from({ length: nClasses }, () => new Array<number>(this.featureCount).fill(0));
    this.interceptVector_ = new Array<number>(nClasses).fill(0);

    this.nIter_ = this.maxIter;
    for (let iter = 0; iter < this.maxIter; iter += 1) {
      let maxUpdate = 0;
      for (let i = 0; i < X.length; i += 1) {
        const scores = this.coefMatrix_.map((coef, c) => dot(X[i], coef) + this.interceptVector_[c]);
        const predClass = argmax(scores);
        const trueClass = this.classes_.indexOf(y[i]);
        if (predClass === trueClass) {
          continue;
        }

        const margin = scores[trueClass] - scores[predClass];
        const hingeLoss = Math.max(0, 1 - margin);
        if (hingeLoss <= 0) {
          continue;
        }

        let normSq = 0;
        for (let f = 0; f < this.featureCount; f += 1) {
          normSq += 2 * X[i][f] * X[i][f];
        }
        if (this.fitIntercept) {
          normSq += 2;
        }
        const denominator = normSq + (this.loss === "squared_hinge" ? 1 / (2 * this.C) : 0);
        const tau = Math.min(this.C, hingeLoss / Math.max(denominator, 1e-12));

        for (let f = 0; f < this.featureCount; f += 1) {
          const delta = tau * X[i][f];
          this.coefMatrix_[trueClass][f] += delta;
          this.coefMatrix_[predClass][f] -= delta;
          maxUpdate = Math.max(maxUpdate, Math.abs(delta));
        }
        if (this.fitIntercept) {
          this.interceptVector_[trueClass] += tau;
          this.interceptVector_[predClass] -= tau;
          maxUpdate = Math.max(maxUpdate, Math.abs(tau));
        }
      }

      if (maxUpdate <= this.tolerance) {
        this.nIter_ = iter + 1;
        break;
      }
    }

    if (nClasses === 2) {
      this.coef_ = this.coefMatrix_[1].slice();
      this.intercept_ = this.interceptVector_[1];
    } else {
      this.coef_ = this.coefMatrix_.map((row) => row.slice());
      this.intercept_ = this.interceptVector_.slice();
    }

    this.fitted = true;
    return this;
  }

  decisionFunction(X: Matrix): Matrix {
    this.assertFitted();
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.featureCount) {
      throw new Error(`Feature size mismatch. Expected ${this.featureCount}, got ${X[0].length}.`);
    }
    return X.map((row) => this.coefMatrix_.map((coef, c) => dot(row, coef) + this.interceptVector_[c]));
  }

  predict(X: Matrix): Vector {
    return this.decisionFunction(X).map((scores) => this.classes_[argmax(scores)]);
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return accuracyScore(y, this.predict(X));
  }

  private validateOptions(): void {
    if (!Number.isFinite(this.C) || this.C <= 0) {
      throw new Error(`C must be finite and > 0. Got ${this.C}.`);
    }
    if (!Number.isInteger(this.maxIter) || this.maxIter < 1) {
      throw new Error(`maxIter must be an integer >= 1. Got ${this.maxIter}.`);
    }
    if (!Number.isFinite(this.tolerance) || this.tolerance < 0) {
      throw new Error(`tolerance must be finite and >= 0. Got ${this.tolerance}.`);
    }
  }

  private assertFitted(): void {
    if (!this.fitted) {
      throw new Error("PassiveAggressiveClassifier has not been fitted.");
    }
  }
}