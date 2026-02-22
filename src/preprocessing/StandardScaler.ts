import type { Matrix, Vector } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";

export class StandardScaler {
  mean_: Vector | null = null;
  scale_: Vector | null = null;

  fit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    const nSamples = X.length;
    const nFeatures = X[0].length;
    const means = new Array(nFeatures).fill(0);
    const variances = new Array(nFeatures).fill(0);

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

    const scales = variances.map((v) => {
      const std = Math.sqrt(v / nSamples);
      return std === 0 ? 1 : std;
    });

    this.mean_ = means;
    this.scale_ = scales;
    return this;
  }

  transform(X: Matrix): Matrix {
    if (!this.mean_ || !this.scale_) {
      throw new Error("StandardScaler has not been fitted.");
    }

    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    if (X[0].length !== this.mean_.length) {
      throw new Error(
        `Feature size mismatch. Expected ${this.mean_.length}, got ${X[0].length}.`,
      );
    }

    return X.map((row) =>
      row.map((value, featureIdx) => (value - this.mean_![featureIdx]) / this.scale_![featureIdx]),
    );
  }

  fitTransform(X: Matrix): Matrix {
    return this.fit(X).transform(X);
  }

  inverseTransform(X: Matrix): Matrix {
    if (!this.mean_ || !this.scale_) {
      throw new Error("StandardScaler has not been fitted.");
    }

    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    if (X[0].length !== this.mean_.length) {
      throw new Error(
        `Feature size mismatch. Expected ${this.mean_.length}, got ${X[0].length}.`,
      );
    }

    return X.map((row) =>
      row.map((value, featureIdx) => value * this.scale_![featureIdx] + this.mean_![featureIdx]),
    );
  }
}
