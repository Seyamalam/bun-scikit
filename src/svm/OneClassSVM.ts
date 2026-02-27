import type { Matrix, Vector } from "../types";
import { dot } from "../utils/linalg";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";
import { type GammaOption } from "./kernelUtils";

export type OneClassSVMKernel = "rbf" | "linear";

export interface OneClassSVMOptions {
  nu?: number;
  kernel?: OneClassSVMKernel;
  gamma?: GammaOption;
}

function euclideanSquared(a: Vector, b: Vector): number {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

function quantile(values: Vector, q: number): number {
  if (values.length === 0) {
    return 0;
  }
  const sorted = values.slice().sort((a, b) => a - b);
  const pos = Math.max(0, Math.min(sorted.length - 1, q * (sorted.length - 1)));
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) {
    return sorted[lo];
  }
  const alpha = pos - lo;
  return sorted[lo] * (1 - alpha) + sorted[hi] * alpha;
}

function columnVariance(X: Matrix): number {
  const nSamples = X.length;
  const nFeatures = X[0].length;
  let total = 0;
  for (let feature = 0; feature < nFeatures; feature += 1) {
    let mean = 0;
    for (let i = 0; i < nSamples; i += 1) {
      mean += X[i][feature];
    }
    mean /= nSamples;
    let varSum = 0;
    for (let i = 0; i < nSamples; i += 1) {
      const d = X[i][feature] - mean;
      varSum += d * d;
    }
    total += varSum / Math.max(1, nSamples);
  }
  return total / Math.max(1, nFeatures);
}

function resolveGammaValue(gamma: GammaOption, nFeatures: number, variance: number): number {
  if (typeof gamma === "number") {
    return gamma;
  }
  if (gamma === "auto") {
    return 1 / nFeatures;
  }
  return 1 / (nFeatures * variance);
}

export class OneClassSVM {
  offset_ = 0;
  nFeaturesIn_: number | null = null;

  private nu: number;
  private kernel: OneClassSVMKernel;
  private gamma: GammaOption;
  private gammaValue = 1;
  private center_: Vector | null = null;
  private threshold_ = 0;
  private fitted = false;

  constructor(options: OneClassSVMOptions = {}) {
    this.nu = options.nu ?? 0.5;
    this.kernel = options.kernel ?? "rbf";
    this.gamma = options.gamma ?? "scale";

    if (!Number.isFinite(this.nu) || this.nu <= 0 || this.nu > 1) {
      throw new Error(`nu must be in (0, 1]. Got ${this.nu}.`);
    }
  }

  fit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    const nSamples = X.length;
    const nFeatures = X[0].length;
    const center = new Array<number>(nFeatures).fill(0);
    for (let i = 0; i < nSamples; i += 1) {
      for (let j = 0; j < nFeatures; j += 1) {
        center[j] += X[i][j];
      }
    }
    for (let j = 0; j < nFeatures; j += 1) {
      center[j] /= nSamples;
    }

    const variance = Math.max(1e-12, columnVariance(X));
    this.gammaValue = resolveGammaValue(this.gamma, nFeatures, variance);
    const trainScores = X.map((row) => this.rawScore(row, center));
    this.threshold_ = quantile(trainScores, this.nu);
    this.offset_ = this.threshold_;
    this.center_ = center;
    this.nFeaturesIn_ = nFeatures;
    this.fitted = true;
    return this;
  }

  scoreSamples(X: Matrix): Vector {
    this.assertFitted();
    this.validatePredictInput(X);
    return X.map((row) => this.rawScore(row, this.center_!));
  }

  decisionFunction(X: Matrix): Vector {
    return this.scoreSamples(X).map((score) => score - this.offset_);
  }

  predict(X: Matrix): Vector {
    return this.decisionFunction(X).map((score) => (score >= 0 ? 1 : -1));
  }

  private rawScore(row: Vector, center: Vector): number {
    if (this.kernel === "linear") {
      return dot(row, center);
    }
    return Math.exp(-this.gammaValue * euclideanSquared(row, center));
  }

  private validatePredictInput(X: Matrix): void {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }
  }

  private assertFitted(): void {
    if (!this.fitted || !this.center_ || this.nFeaturesIn_ === null) {
      throw new Error("OneClassSVM has not been fitted.");
    }
  }
}
