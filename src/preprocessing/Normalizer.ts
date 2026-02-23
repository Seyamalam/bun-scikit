import type { Matrix } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";

export interface NormalizerOptions {
  norm?: "l1" | "l2" | "max";
}

export class Normalizer {
  private readonly norm: "l1" | "l2" | "max";
  private nFeatures_: number | null = null;

  constructor(options: NormalizerOptions = {}) {
    this.norm = options.norm ?? "l2";
  }

  fit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    this.nFeatures_ = X[0].length;
    return this;
  }

  transform(X: Matrix): Matrix {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (this.nFeatures_ !== null && X[0].length !== this.nFeatures_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeatures_}, got ${X[0].length}.`);
    }

    return X.map((row) => {
      let scale = 0;
      if (this.norm === "l1") {
        for (let i = 0; i < row.length; i += 1) {
          scale += Math.abs(row[i]);
        }
      } else if (this.norm === "l2") {
        for (let i = 0; i < row.length; i += 1) {
          scale += row[i] * row[i];
        }
        scale = Math.sqrt(scale);
      } else {
        for (let i = 0; i < row.length; i += 1) {
          const abs = Math.abs(row[i]);
          if (abs > scale) {
            scale = abs;
          }
        }
      }

      if (scale === 0) {
        return [...row];
      }
      return row.map((value) => value / scale);
    });
  }

  fitTransform(X: Matrix): Matrix {
    return this.fit(X).transform(X);
  }
}
