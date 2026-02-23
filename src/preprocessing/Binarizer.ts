import type { Matrix } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";

export interface BinarizerOptions {
  threshold?: number;
}

export class Binarizer {
  nFeaturesIn_: number | null = null;
  private readonly threshold: number;

  constructor(options: BinarizerOptions = {}) {
    this.threshold = options.threshold ?? 0;
    if (!Number.isFinite(this.threshold)) {
      throw new Error(`threshold must be finite. Got ${this.threshold}.`);
    }
  }

  fit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    this.nFeaturesIn_ = X[0].length;
    return this;
  }

  transform(X: Matrix): Matrix {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    if (this.nFeaturesIn_ !== null && X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }

    return X.map((row) => row.map((value) => (value > this.threshold ? 1 : 0)));
  }

  fitTransform(X: Matrix): Matrix {
    return this.fit(X).transform(X);
  }
}
