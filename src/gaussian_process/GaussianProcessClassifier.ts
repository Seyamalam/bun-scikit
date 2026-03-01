import type { ClassificationModel, Matrix, Vector } from "../types";
import { accuracyScore } from "../metrics/classification";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  validateClassificationInputs,
} from "../utils/validation";
import { argmax, normalizeProbabilitiesInPlace, uniqueSortedLabels } from "../utils/classification";
import { inverseMatrix, multiplyMatrixVector } from "../utils/linalg";
import { kernelMatrix } from "./shared";

export interface GaussianProcessClassifierOptions {
  alpha?: number;
  lengthScale?: number;
  maxIter?: number;
}

function invertWithJitter(K: Matrix): Matrix {
  let jitter = 1e-10;
  for (let attempt = 0; attempt < 8; attempt += 1) {
    try {
      const regularized = K.map((row) => row.slice());
      for (let i = 0; i < regularized.length; i += 1) {
        regularized[i][i] += jitter;
      }
      return inverseMatrix(regularized);
    } catch {
      jitter *= 10;
    }
  }
  throw new Error("GaussianProcessClassifier could not invert covariance matrix.");
}

export class GaussianProcessClassifier implements ClassificationModel {
  classes_: Vector = [];
  XTrain_: Matrix | null = null;
  nFeaturesIn_: number | null = null;

  private alpha: number;
  private lengthScale: number;
  private maxIter: number;
  private dualCoef_: Matrix | null = null;
  private KInv: Matrix | null = null;
  private fitted = false;

  constructor(options: GaussianProcessClassifierOptions = {}) {
    this.alpha = options.alpha ?? 1e-6;
    this.lengthScale = options.lengthScale ?? 1;
    this.maxIter = options.maxIter ?? 100;

    if (!Number.isFinite(this.alpha) || this.alpha < 0) {
      throw new Error(`alpha must be finite and >= 0. Got ${this.alpha}.`);
    }
    if (!Number.isFinite(this.lengthScale) || this.lengthScale <= 0) {
      throw new Error(`lengthScale must be finite and > 0. Got ${this.lengthScale}.`);
    }
    if (!Number.isInteger(this.maxIter) || this.maxIter < 1) {
      throw new Error(`maxIter must be an integer >= 1. Got ${this.maxIter}.`);
    }
  }

  fit(X: Matrix, y: Vector): this {
    validateClassificationInputs(X, y);
    const classes = uniqueSortedLabels(y);
    if (classes.length < 2) {
      throw new Error("GaussianProcessClassifier requires at least two classes.");
    }

    const K = kernelMatrix(X, X, this.lengthScale);
    for (let i = 0; i < K.length; i += 1) {
      K[i][i] += this.alpha;
    }
    const KInv = invertWithJitter(K);

    const dualCoef: Matrix = Array.from({ length: classes.length }, () => new Array<number>(X.length).fill(0));
    for (let c = 0; c < classes.length; c += 1) {
      const targets = new Array<number>(X.length).fill(-1);
      for (let i = 0; i < y.length; i += 1) {
        if (y[i] === classes[c]) {
          targets[i] = 1;
        }
      }
      dualCoef[c] = multiplyMatrixVector(KInv, targets);
    }

    this.classes_ = classes;
    this.XTrain_ = X.map((row) => row.slice());
    this.nFeaturesIn_ = X[0].length;
    this.KInv = KInv;
    this.dualCoef_ = dualCoef;
    this.fitted = true;
    return this;
  }

  predictProba(X: Matrix): Matrix {
    this.assertFitted();
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }

    const KStar = kernelMatrix(X, this.XTrain_!, this.lengthScale);
    const out: Matrix = new Array(X.length);

    for (let i = 0; i < X.length; i += 1) {
      const logits = new Array<number>(this.classes_.length).fill(0);
      for (let c = 0; c < this.classes_.length; c += 1) {
        let score = 0;
        for (let j = 0; j < this.XTrain_!.length; j += 1) {
          score += KStar[i][j] * this.dualCoef_![c][j];
        }
        logits[c] = score;
      }
      let maxValue = logits[0];
      for (let c = 1; c < logits.length; c += 1) {
        if (logits[c] > maxValue) {
          maxValue = logits[c];
        }
      }
      for (let c = 0; c < logits.length; c += 1) {
        logits[c] = Math.exp(logits[c] - maxValue);
      }
      normalizeProbabilitiesInPlace(logits);
      out[i] = logits;
    }

    return out;
  }

  predict(X: Matrix): Vector {
    const probabilities = this.predictProba(X);
    return probabilities.map((row) => this.classes_[argmax(row)]);
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return accuracyScore(y, this.predict(X));
  }

  private assertFitted(): void {
    if (
      !this.fitted ||
      this.classes_.length === 0 ||
      this.XTrain_ === null ||
      this.dualCoef_ === null ||
      this.KInv === null ||
      this.nFeaturesIn_ === null
    ) {
      throw new Error("GaussianProcessClassifier has not been fitted.");
    }
  }
}

