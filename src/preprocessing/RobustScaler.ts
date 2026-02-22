import type { Matrix, Vector } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";

export interface RobustScalerOptions {
  withCentering?: boolean;
  withScaling?: boolean;
  quantileRange?: [number, number];
}

function percentile(sortedValues: number[], q: number): number {
  if (sortedValues.length === 0) {
    throw new Error("Cannot compute percentile of an empty array.");
  }
  if (q <= 0) {
    return sortedValues[0];
  }
  if (q >= 100) {
    return sortedValues[sortedValues.length - 1];
  }

  const position = (q / 100) * (sortedValues.length - 1);
  const lowerIndex = Math.floor(position);
  const upperIndex = Math.ceil(position);
  if (lowerIndex === upperIndex) {
    return sortedValues[lowerIndex];
  }

  const weight = position - lowerIndex;
  return sortedValues[lowerIndex] * (1 - weight) + sortedValues[upperIndex] * weight;
}

export class RobustScaler {
  center_: Vector | null = null;
  scale_: Vector | null = null;

  private readonly withCentering: boolean;
  private readonly withScaling: boolean;
  private readonly quantileRange: [number, number];

  constructor(options: RobustScalerOptions = {}) {
    this.withCentering = options.withCentering ?? true;
    this.withScaling = options.withScaling ?? true;
    this.quantileRange = options.quantileRange ?? [25, 75];

    const [qMin, qMax] = this.quantileRange;
    if (
      !Number.isFinite(qMin) ||
      !Number.isFinite(qMax) ||
      qMin < 0 ||
      qMax > 100 ||
      qMin >= qMax
    ) {
      throw new Error(
        `quantileRange must satisfy 0 <= qMin < qMax <= 100. Got [${qMin}, ${qMax}].`,
      );
    }
  }

  fit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    const nFeatures = X[0].length;
    const centers = new Array<number>(nFeatures).fill(0);
    const scales = new Array<number>(nFeatures).fill(1);
    const [qMin, qMax] = this.quantileRange;

    for (let featureIndex = 0; featureIndex < nFeatures; featureIndex += 1) {
      const values = new Array<number>(X.length);
      for (let i = 0; i < X.length; i += 1) {
        values[i] = X[i][featureIndex];
      }
      values.sort((a, b) => a - b);

      centers[featureIndex] = percentile(values, 50);
      const lower = percentile(values, qMin);
      const upper = percentile(values, qMax);
      const iqr = upper - lower;
      scales[featureIndex] = iqr === 0 ? 1 : iqr;
    }

    this.center_ = centers;
    this.scale_ = scales;
    return this;
  }

  transform(X: Matrix): Matrix {
    if (!this.center_ || !this.scale_) {
      throw new Error("RobustScaler has not been fitted.");
    }

    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    if (X[0].length !== this.center_.length) {
      throw new Error(`Feature size mismatch. Expected ${this.center_.length}, got ${X[0].length}.`);
    }

    return X.map((row) =>
      row.map((value, featureIndex) => {
        let out = value;
        if (this.withCentering) {
          out -= this.center_![featureIndex];
        }
        if (this.withScaling) {
          out /= this.scale_![featureIndex];
        }
        return out;
      }),
    );
  }

  fitTransform(X: Matrix): Matrix {
    return this.fit(X).transform(X);
  }

  inverseTransform(X: Matrix): Matrix {
    if (!this.center_ || !this.scale_) {
      throw new Error("RobustScaler has not been fitted.");
    }

    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    if (X[0].length !== this.center_.length) {
      throw new Error(`Feature size mismatch. Expected ${this.center_.length}, got ${X[0].length}.`);
    }

    return X.map((row) =>
      row.map((value, featureIndex) => {
        let out = value;
        if (this.withScaling) {
          out *= this.scale_![featureIndex];
        }
        if (this.withCentering) {
          out += this.center_![featureIndex];
        }
        return out;
      }),
    );
  }
}
