import type { Matrix } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";

export interface VarianceThresholdOptions {
  threshold?: number;
}

export class VarianceThreshold {
  variances_: number[] | null = null;
  nFeaturesIn_: number | null = null;
  selectedFeatureIndices_: number[] | null = null;

  private readonly threshold: number;

  constructor(options: VarianceThresholdOptions = {}) {
    this.threshold = options.threshold ?? 0;
    if (!Number.isFinite(this.threshold) || this.threshold < 0) {
      throw new Error(`threshold must be finite and >= 0. Got ${this.threshold}.`);
    }
  }

  fit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    const nSamples = X.length;
    const nFeatures = X[0].length;
    const means = new Array<number>(nFeatures).fill(0);
    const variances = new Array<number>(nFeatures).fill(0);

    for (let i = 0; i < nSamples; i += 1) {
      for (let j = 0; j < nFeatures; j += 1) {
        means[j] += X[i][j];
      }
    }
    for (let j = 0; j < nFeatures; j += 1) {
      means[j] /= nSamples;
    }
    for (let i = 0; i < nSamples; i += 1) {
      for (let j = 0; j < nFeatures; j += 1) {
        const diff = X[i][j] - means[j];
        variances[j] += diff * diff;
      }
    }
    for (let j = 0; j < nFeatures; j += 1) {
      variances[j] /= nSamples;
    }

    const selectedFeatureIndices: number[] = [];
    for (let j = 0; j < nFeatures; j += 1) {
      if (variances[j] > this.threshold) {
        selectedFeatureIndices.push(j);
      }
    }
    if (selectedFeatureIndices.length === 0) {
      throw new Error("No feature in X meets the variance threshold.");
    }

    this.nFeaturesIn_ = nFeatures;
    this.variances_ = variances;
    this.selectedFeatureIndices_ = selectedFeatureIndices;
    return this;
  }

  transform(X: Matrix): Matrix {
    if (!this.selectedFeatureIndices_ || this.nFeaturesIn_ === null) {
      throw new Error("VarianceThreshold has not been fitted.");
    }
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }

    return X.map((row) => this.selectedFeatureIndices_!.map((featureIdx) => row[featureIdx]));
  }

  fitTransform(X: Matrix): Matrix {
    return this.fit(X).transform(X);
  }
}
