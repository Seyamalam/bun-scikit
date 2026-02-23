import type { Matrix, Vector } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";

export class MaxAbsScaler {
  maxAbs_: Vector | null = null;

  fit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    const nFeatures = X[0].length;
    const maxAbs = new Array<number>(nFeatures).fill(0);

    for (let i = 0; i < X.length; i += 1) {
      for (let j = 0; j < nFeatures; j += 1) {
        const abs = Math.abs(X[i][j]);
        if (abs > maxAbs[j]) {
          maxAbs[j] = abs;
        }
      }
    }

    for (let j = 0; j < nFeatures; j += 1) {
      if (maxAbs[j] === 0) {
        maxAbs[j] = 1;
      }
    }

    this.maxAbs_ = maxAbs;
    return this;
  }

  transform(X: Matrix): Matrix {
    if (!this.maxAbs_) {
      throw new Error("MaxAbsScaler has not been fitted.");
    }

    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    if (X[0].length !== this.maxAbs_.length) {
      throw new Error(
        `Feature size mismatch. Expected ${this.maxAbs_.length}, got ${X[0].length}.`,
      );
    }

    return X.map((row) => row.map((value, featureIdx) => value / this.maxAbs_![featureIdx]));
  }

  fitTransform(X: Matrix): Matrix {
    return this.fit(X).transform(X);
  }

  inverseTransform(X: Matrix): Matrix {
    if (!this.maxAbs_) {
      throw new Error("MaxAbsScaler has not been fitted.");
    }

    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    if (X[0].length !== this.maxAbs_.length) {
      throw new Error(
        `Feature size mismatch. Expected ${this.maxAbs_.length}, got ${X[0].length}.`,
      );
    }

    return X.map((row) => row.map((value, featureIdx) => value * this.maxAbs_![featureIdx]));
  }
}
