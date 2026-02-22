import type { Matrix, Vector } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";

export interface MinMaxScalerOptions {
  featureRange?: [number, number];
}

export class MinMaxScaler {
  dataMin_: Vector | null = null;
  dataMax_: Vector | null = null;
  dataRange_: Vector | null = null;
  scale_: Vector | null = null;
  min_: Vector | null = null;

  private readonly featureRange: [number, number];

  constructor(options: MinMaxScalerOptions = {}) {
    this.featureRange = options.featureRange ?? [0, 1];
    const [rangeMin, rangeMax] = this.featureRange;
    if (!Number.isFinite(rangeMin) || !Number.isFinite(rangeMax) || rangeMin >= rangeMax) {
      throw new Error(
        `featureRange must be finite and satisfy min < max. Got [${rangeMin}, ${rangeMax}].`,
      );
    }
  }

  fit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    const nFeatures = X[0].length;
    const dataMin = new Array(nFeatures).fill(Number.POSITIVE_INFINITY);
    const dataMax = new Array(nFeatures).fill(Number.NEGATIVE_INFINITY);

    for (let i = 0; i < X.length; i += 1) {
      for (let j = 0; j < nFeatures; j += 1) {
        const value = X[i][j];
        if (value < dataMin[j]) {
          dataMin[j] = value;
        }
        if (value > dataMax[j]) {
          dataMax[j] = value;
        }
      }
    }

    const [rangeMin, rangeMax] = this.featureRange;
    const targetRange = rangeMax - rangeMin;
    const dataRange = new Array(nFeatures).fill(0);
    const scale = new Array(nFeatures).fill(1);
    const min = new Array(nFeatures).fill(rangeMin);

    for (let j = 0; j < nFeatures; j += 1) {
      const featureDataRange = dataMax[j] - dataMin[j];
      dataRange[j] = featureDataRange;
      const safeDenominator = featureDataRange === 0 ? 1 : featureDataRange;
      scale[j] = targetRange / safeDenominator;
      min[j] = rangeMin - dataMin[j] * scale[j];
    }

    this.dataMin_ = dataMin;
    this.dataMax_ = dataMax;
    this.dataRange_ = dataRange;
    this.scale_ = scale;
    this.min_ = min;
    return this;
  }

  transform(X: Matrix): Matrix {
    if (!this.scale_ || !this.min_) {
      throw new Error("MinMaxScaler has not been fitted.");
    }

    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    if (X[0].length !== this.scale_.length) {
      throw new Error(`Feature size mismatch. Expected ${this.scale_.length}, got ${X[0].length}.`);
    }

    return X.map((row) =>
      row.map((value, featureIdx) => value * this.scale_![featureIdx] + this.min_![featureIdx]),
    );
  }

  fitTransform(X: Matrix): Matrix {
    return this.fit(X).transform(X);
  }

  inverseTransform(X: Matrix): Matrix {
    if (!this.scale_ || !this.min_) {
      throw new Error("MinMaxScaler has not been fitted.");
    }

    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    if (X[0].length !== this.scale_.length) {
      throw new Error(`Feature size mismatch. Expected ${this.scale_.length}, got ${X[0].length}.`);
    }

    return X.map((row) =>
      row.map((value, featureIdx) => (value - this.min_![featureIdx]) / this.scale_![featureIdx]),
    );
  }
}
