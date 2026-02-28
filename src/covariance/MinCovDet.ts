import type { Matrix, Vector } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";
import {
  covarianceMatrix,
  featureMeans,
  mahalanobisDistanceSquared,
  matrixDeterminant,
  regularizedPrecision,
} from "./shared";

export interface MinCovDetOptions {
  supportFraction?: number;
  maxIter?: number;
}

function subsetRows(X: Matrix, indices: number[]): Matrix {
  return indices.map((idx) => X[idx]);
}

function euclideanDistanceSquared(a: Vector, b: Vector): number {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

export class MinCovDet {
  location_: Vector | null = null;
  covariance_: Matrix | null = null;
  precision_: Matrix | null = null;
  support_: boolean[] | null = null;
  nFeaturesIn_: number | null = null;

  private supportFraction: number;
  private maxIter: number;
  private fitted = false;

  constructor(options: MinCovDetOptions = {}) {
    this.supportFraction = options.supportFraction ?? 0.75;
    this.maxIter = options.maxIter ?? 5;
    if (!Number.isFinite(this.supportFraction) || this.supportFraction <= 0 || this.supportFraction > 1) {
      throw new Error(`supportFraction must be in (0, 1]. Got ${this.supportFraction}.`);
    }
    if (!Number.isInteger(this.maxIter) || this.maxIter < 1) {
      throw new Error(`maxIter must be an integer >= 1. Got ${this.maxIter}.`);
    }
  }

  fit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    const nSamples = X.length;
    const h = Math.max(2, Math.floor(nSamples * this.supportFraction));
    let center = featureMeans(X);
    let supportIndices = Array.from({ length: nSamples }, (_, idx) => idx);

    for (let iter = 0; iter < this.maxIter; iter += 1) {
      const distances = X.map((row, idx) => ({
        idx,
        dist: euclideanDistanceSquared(row, center),
      })).sort((a, b) => a.dist - b.dist);
      supportIndices = distances.slice(0, h).map((item) => item.idx);
      const subset = subsetRows(X, supportIndices);
      center = featureMeans(subset);
    }

    const supportSet = new Set(supportIndices);
    const support = new Array<boolean>(nSamples).fill(false);
    for (let i = 0; i < support.length; i += 1) {
      support[i] = supportSet.has(i);
    }

    const subset = subsetRows(X, supportIndices);
    const covariance = covarianceMatrix(subset, center);
    const precision = regularizedPrecision(covariance);

    this.location_ = center;
    this.covariance_ = covariance;
    this.precision_ = precision;
    this.support_ = support;
    this.nFeaturesIn_ = X[0].length;
    this.fitted = true;
    return this;
  }

  mahalanobis(X: Matrix): Vector {
    this.assertFitted();
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }
    return X.map((row) =>
      Math.sqrt(mahalanobisDistanceSquared(row, this.location_!, this.precision_!)),
    );
  }

  scoreSamples(X: Matrix): Vector {
    this.assertFitted();
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }
    const det = Math.max(1e-12, matrixDeterminant(this.covariance_!));
    const logDet = Math.log(det);
    const d = this.nFeaturesIn_!;
    return X.map((row) => {
      const mah = mahalanobisDistanceSquared(row, this.location_!, this.precision_!);
      return -0.5 * (d * Math.log(2 * Math.PI) + logDet + mah);
    });
  }

  score(X: Matrix): number {
    const scores = this.scoreSamples(X);
    let sum = 0;
    for (let i = 0; i < scores.length; i += 1) {
      sum += scores[i];
    }
    return sum / scores.length;
  }

  private assertFitted(): void {
    if (
      !this.fitted ||
      !this.location_ ||
      !this.covariance_ ||
      !this.precision_ ||
      !this.support_ ||
      this.nFeaturesIn_ === null
    ) {
      throw new Error("MinCovDet has not been fitted.");
    }
  }
}
