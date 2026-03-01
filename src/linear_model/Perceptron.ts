import type { ClassificationModel, Matrix, Vector } from "../types";
import { accuracyScore } from "../metrics/classification";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  validateClassificationInputs,
} from "../utils/validation";
import { argmax, uniqueSortedLabels } from "../utils/classification";
import { dot } from "../utils/linalg";

export interface PerceptronOptions {
  alpha?: number;
  fitIntercept?: boolean;
  maxIter?: number;
  tolerance?: number;
  shuffle?: boolean;
  randomState?: number;
}

function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state ^ (state >>> 15);
    t = Math.imul(t, state | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class Perceptron implements ClassificationModel {
  coef_: Vector | Matrix = [];
  intercept_: number | Vector = 0;
  classes_: Vector = [];
  nIter_ = 0;

  private alpha: number;
  private fitIntercept: boolean;
  private maxIter: number;
  private tolerance: number;
  private shuffle: boolean;
  private randomState: number;
  private coefMatrix_: Matrix = [];
  private interceptVector_: Vector = [];
  private fitted = false;
  private featureCount = 0;

  constructor(options: PerceptronOptions = {}) {
    this.alpha = options.alpha ?? 0.0001;
    this.fitIntercept = options.fitIntercept ?? true;
    this.maxIter = options.maxIter ?? 1000;
    this.tolerance = options.tolerance ?? 1e-4;
    this.shuffle = options.shuffle ?? true;
    this.randomState = options.randomState ?? 42;
    this.validateOptions();
  }

  fit(X: Matrix, y: Vector, sampleWeight?: Vector): this {
    validateClassificationInputs(X, y);
    this.classes_ = uniqueSortedLabels(y);
    this.featureCount = X[0].length;

    const nClasses = this.classes_.length;
    const nSamples = X.length;
    this.coefMatrix_ = Array.from({ length: nClasses }, () => new Array<number>(this.featureCount).fill(0));
    this.interceptVector_ = new Array<number>(nClasses).fill(0);

    const order = Array.from({ length: nSamples }, (_, i) => i);
    const rng = mulberry32(this.randomState);

    this.nIter_ = this.maxIter;
    for (let iter = 0; iter < this.maxIter; iter += 1) {
      if (this.shuffle) {
        for (let i = order.length - 1; i > 0; i -= 1) {
          const j = Math.floor(rng() * (i + 1));
          const t = order[i];
          order[i] = order[j];
          order[j] = t;
        }
      }

      let mistakes = 0;
      for (let idx = 0; idx < order.length; idx += 1) {
        const i = order[idx];
        const scores = new Array<number>(nClasses);
        for (let c = 0; c < nClasses; c += 1) {
          scores[c] = dot(X[i], this.coefMatrix_[c]) + this.interceptVector_[c];
        }
        const predClass = argmax(scores);
        const trueClass = this.classes_.indexOf(y[i]);
        if (predClass === trueClass) {
          continue;
        }
        mistakes += 1;
        for (let f = 0; f < this.featureCount; f += 1) {
          this.coefMatrix_[trueClass][f] += X[i][f];
          this.coefMatrix_[predClass][f] -= X[i][f];
          this.coefMatrix_[trueClass][f] -= this.alpha * this.coefMatrix_[trueClass][f];
          this.coefMatrix_[predClass][f] -= this.alpha * this.coefMatrix_[predClass][f];
        }
        if (this.fitIntercept) {
          this.interceptVector_[trueClass] += 1;
          this.interceptVector_[predClass] -= 1;
        }
      }

      const errorRate = mistakes / nSamples;
      if (errorRate <= this.tolerance) {
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

  predict(X: Matrix): Vector {
    this.assertFitted();
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.featureCount) {
      throw new Error(`Feature size mismatch. Expected ${this.featureCount}, got ${X[0].length}.`);
    }
    return X.map((row) => {
      const scores = this.coefMatrix_.map((coef, c) => dot(row, coef) + this.interceptVector_[c]);
      return this.classes_[argmax(scores)];
    });
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return accuracyScore(y, this.predict(X));
  }

  private assertFitted(): void {
    if (!this.fitted) {
      throw new Error("Perceptron has not been fitted.");
    }
  }

  private validateOptions(): void {
    if (!Number.isFinite(this.alpha) || this.alpha < 0) {
      throw new Error(`alpha must be finite and >= 0. Got ${this.alpha}.`);
    }
    if (!Number.isInteger(this.maxIter) || this.maxIter < 1) {
      throw new Error(`maxIter must be an integer >= 1. Got ${this.maxIter}.`);
    }
    if (!Number.isFinite(this.tolerance) || this.tolerance < 0) {
      throw new Error(`tolerance must be finite and >= 0. Got ${this.tolerance}.`);
    }
  }
}