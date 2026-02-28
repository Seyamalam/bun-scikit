import type { Matrix, Vector } from "../types";
import {
  assertConsistentRowSize,
  assertNonEmptyMatrix,
} from "../utils/validation";

export type MissingIndicatorFeatures = "missing-only" | "all";

export interface MissingIndicatorOptions {
  features?: MissingIndicatorFeatures;
  errorOnNew?: boolean;
}

function isMissing(value: number): boolean {
  return Number.isNaN(value);
}

function assertFiniteOrMissing(X: Matrix, label = "X"): void {
  for (let i = 0; i < X.length; i += 1) {
    for (let j = 0; j < X[i].length; j += 1) {
      const value = X[i][j];
      if (!Number.isFinite(value) && !isMissing(value)) {
        throw new Error(`${label} contains non-finite non-missing value at [${i}, ${j}].`);
      }
    }
  }
}

export class MissingIndicator {
  features_: Vector | null = null;
  nFeaturesIn_: number | null = null;

  private features: MissingIndicatorFeatures;
  private errorOnNew: boolean;
  private missingSet: Set<number> | null = null;
  private fitted = false;

  constructor(options: MissingIndicatorOptions = {}) {
    this.features = options.features ?? "missing-only";
    this.errorOnNew = options.errorOnNew ?? true;
    if (!(this.features === "missing-only" || this.features === "all")) {
      throw new Error(`features must be 'missing-only' or 'all'. Got ${this.features}.`);
    }
  }

  fit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteOrMissing(X);

    const missing = new Set<number>();
    for (let i = 0; i < X.length; i += 1) {
      for (let j = 0; j < X[i].length; j += 1) {
        if (isMissing(X[i][j])) {
          missing.add(j);
        }
      }
    }

    const featureIndices =
      this.features === "all"
        ? Array.from({ length: X[0].length }, (_, idx) => idx)
        : Array.from(missing).sort((a, b) => a - b);

    this.features_ = featureIndices;
    this.missingSet = missing;
    this.nFeaturesIn_ = X[0].length;
    this.fitted = true;
    return this;
  }

  transform(X: Matrix): Matrix {
    this.assertFitted();
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteOrMissing(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }

    if (this.errorOnNew) {
      for (let i = 0; i < X.length; i += 1) {
        for (let j = 0; j < X[i].length; j += 1) {
          if (isMissing(X[i][j]) && !this.missingSet!.has(j)) {
            throw new Error(
              `New missing values found in feature ${j} during transform and errorOnNew=true.`,
            );
          }
        }
      }
    }

    return X.map((row) =>
      this.features_!.map((featureIndex) => (isMissing(row[featureIndex]) ? 1 : 0)),
    );
  }

  fitTransform(X: Matrix): Matrix {
    return this.fit(X).transform(X);
  }

  private assertFitted(): void {
    if (!this.fitted || !this.features_ || !this.missingSet || this.nFeaturesIn_ === null) {
      throw new Error("MissingIndicator has not been fitted.");
    }
  }
}
