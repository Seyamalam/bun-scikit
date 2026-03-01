import type { Matrix, RegressionModel, Vector } from "../types";
import { r2Score } from "../metrics/regression";
import { assertConsistentRowSize, assertFiniteMatrix, assertFiniteVector, validateRegressionInputs } from "../utils/validation";
import { inverseMatrix, multiplyMatrixVector } from "../utils/linalg";
import { kernelMatrix, rbfKernel } from "./shared";

export interface GaussianProcessRegressorOptions {
  alpha?: number | Vector;
  lengthScale?: number;
  normalizeY?: boolean;
}

function mean(values: Vector): number {
  let sum = 0;
  for (let i = 0; i < values.length; i += 1) {
    sum += values[i];
  }
  return sum / values.length;
}

function addNoiseToKernel(K: Matrix, alpha: number | Vector): Matrix {
  const out = K.map((row) => row.slice());
  if (typeof alpha === "number") {
    for (let i = 0; i < out.length; i += 1) {
      out[i][i] += alpha;
    }
    return out;
  }
  if (alpha.length !== out.length) {
    throw new Error(`alpha vector length must match sample count ${out.length}.`);
  }
  for (let i = 0; i < out.length; i += 1) {
    out[i][i] += alpha[i];
  }
  return out;
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
  throw new Error("GaussianProcessRegressor could not invert covariance matrix.");
}

export class GaussianProcessRegressor implements RegressionModel {
  XTrain_: Matrix | null = null;
  yTrain_: Vector | null = null;
  alpha_: Vector | null = null;
  kernelMatrix_: Matrix | null = null;
  nFeaturesIn_: number | null = null;

  private alpha: number | Vector;
  private lengthScale: number;
  private normalizeY: boolean;
  private yMean = 0;
  private KInv: Matrix | null = null;
  private fitted = false;

  constructor(options: GaussianProcessRegressorOptions = {}) {
    this.alpha = options.alpha ?? 1e-10;
    this.lengthScale = options.lengthScale ?? 1;
    this.normalizeY = options.normalizeY ?? false;
    if (!Number.isFinite(this.lengthScale) || this.lengthScale <= 0) {
      throw new Error(`lengthScale must be finite and > 0. Got ${this.lengthScale}.`);
    }
    if (typeof this.alpha === "number" && (!Number.isFinite(this.alpha) || this.alpha < 0)) {
      throw new Error(`alpha must be finite and >= 0. Got ${this.alpha}.`);
    }
  }

  fit(X: Matrix, y: Vector): this {
    validateRegressionInputs(X, y);
    if (typeof this.alpha !== "number") {
      assertFiniteVector(this.alpha, "alpha");
      for (let i = 0; i < this.alpha.length; i += 1) {
        if (this.alpha[i] < 0) {
          throw new Error(`alpha values must be >= 0. Got ${this.alpha[i]}.`);
        }
      }
    }

    this.XTrain_ = X.map((row) => row.slice());
    this.yTrain_ = y.slice();
    this.nFeaturesIn_ = X[0].length;

    const yCentered = y.slice();
    if (this.normalizeY) {
      this.yMean = mean(y);
      for (let i = 0; i < yCentered.length; i += 1) {
        yCentered[i] -= this.yMean;
      }
    } else {
      this.yMean = 0;
    }

    const K = kernelMatrix(this.XTrain_, this.XTrain_, this.lengthScale);
    const noisyK = addNoiseToKernel(K, this.alpha);
    const KInv = invertWithJitter(noisyK);
    const alphaVec = multiplyMatrixVector(KInv, yCentered);

    this.kernelMatrix_ = noisyK;
    this.KInv = KInv;
    this.alpha_ = alphaVec;
    this.fitted = true;
    return this;
  }

  predict(X: Matrix): Vector {
    this.assertFitted();
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }

    const KStar = kernelMatrix(X, this.XTrain_!, this.lengthScale);
    const out = new Array<number>(X.length).fill(0);
    for (let i = 0; i < X.length; i += 1) {
      let sum = 0;
      for (let j = 0; j < this.XTrain_!.length; j += 1) {
        sum += KStar[i][j] * this.alpha_![j];
      }
      out[i] = sum + this.yMean;
    }
    return out;
  }

  predictStd(X: Matrix): Vector {
    this.assertFitted();
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }

    const out = new Array<number>(X.length).fill(0);
    for (let i = 0; i < X.length; i += 1) {
      const kVector = new Array<number>(this.XTrain_!.length).fill(0);
      for (let j = 0; j < this.XTrain_!.length; j += 1) {
        kVector[j] = rbfKernel(X[i], this.XTrain_![j], this.lengthScale);
      }

      const temp = multiplyMatrixVector(this.KInv!, kVector);
      let variance = rbfKernel(X[i], X[i], this.lengthScale);
      let reduction = 0;
      for (let j = 0; j < kVector.length; j += 1) {
        reduction += kVector[j] * temp[j];
      }
      variance -= reduction;
      out[i] = Math.sqrt(Math.max(0, variance));
    }
    return out;
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return r2Score(y, this.predict(X));
  }

  private assertFitted(): void {
    if (
      !this.fitted ||
      this.XTrain_ === null ||
      this.yTrain_ === null ||
      this.alpha_ === null ||
      this.kernelMatrix_ === null ||
      this.KInv === null ||
      this.nFeaturesIn_ === null
    ) {
      throw new Error("GaussianProcessRegressor has not been fitted.");
    }
  }
}

