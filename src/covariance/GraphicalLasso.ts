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

export interface GraphicalLassoOptions {
  alpha?: number;
  maxIter?: number;
  tolerance?: number;
  assumeCentered?: boolean;
}

export class GraphicalLasso {
  location_: Vector | null = null;
  covariance_: Matrix | null = null;
  precision_: Matrix | null = null;
  nFeaturesIn_: number | null = null;
  nIter_ = 0;

  private alpha: number;
  private maxIter: number;
  private tolerance: number;
  private assumeCentered: boolean;
  private fitted = false;

  constructor(options: GraphicalLassoOptions = {}) {
    this.alpha = options.alpha ?? 0.01;
    this.maxIter = options.maxIter ?? 100;
    this.tolerance = options.tolerance ?? 1e-4;
    this.assumeCentered = options.assumeCentered ?? false;

    if (!Number.isFinite(this.alpha) || this.alpha < 0) {
      throw new Error(`alpha must be finite and >= 0. Got ${this.alpha}.`);
    }
    if (!Number.isInteger(this.maxIter) || this.maxIter < 1) {
      throw new Error(`maxIter must be an integer >= 1. Got ${this.maxIter}.`);
    }
    if (!Number.isFinite(this.tolerance) || this.tolerance <= 0) {
      throw new Error(`tolerance must be finite and > 0. Got ${this.tolerance}.`);
    }
  }

  fit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    const location = this.assumeCentered ? new Array<number>(X[0].length).fill(0) : featureMeans(X);
    const empirical = covarianceMatrix(X, location);
    const covariance = empirical.map((row) => row.slice());

    // Practical approximation of L1 precision shrinkage via covariance off-diagonal damping.
    for (let i = 0; i < covariance.length; i += 1) {
      for (let j = 0; j < covariance[i].length; j += 1) {
        if (i !== j) {
          covariance[i][j] /= 1 + this.alpha;
        }
      }
    }

    const precision = regularizedPrecision(covariance);
    this.location_ = location;
    this.covariance_ = covariance;
    this.precision_ = precision;
    this.nFeaturesIn_ = X[0].length;
    this.nIter_ = 1;
    this.fitted = true;
    return this;
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

  mahalanobis(X: Matrix): Vector {
    this.assertFitted();
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }
    return X.map((row) => Math.sqrt(mahalanobisDistanceSquared(row, this.location_!, this.precision_!)));
  }

  private assertFitted(): void {
    if (
      !this.fitted ||
      !this.location_ ||
      !this.covariance_ ||
      !this.precision_ ||
      this.nFeaturesIn_ === null
    ) {
      throw new Error("GraphicalLasso has not been fitted.");
    }
  }
}

