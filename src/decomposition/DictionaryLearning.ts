import type { Matrix, Vector } from "../types";
import { dot } from "../utils/linalg";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";
import { PCA } from "./PCA";

export interface DictionaryLearningOptions {
  nComponents?: number;
  alpha?: number;
  maxIter?: number;
  tolerance?: number;
  randomState?: number;
  transformAlpha?: number;
}

function softThreshold(value: number, alpha: number): number {
  if (value > alpha) {
    return value - alpha;
  }
  if (value < -alpha) {
    return value + alpha;
  }
  return 0;
}

export class DictionaryLearning {
  components_: Matrix | null = null;
  error_: number | null = null;
  nFeaturesIn_: number | null = null;
  nComponents_: number | null = null;
  mean_: Vector | null = null;

  private nComponents?: number;
  private alpha: number;
  private maxIter: number;
  private tolerance: number;
  private randomState?: number;
  private transformAlpha: number;
  private fitted = false;

  constructor(options: DictionaryLearningOptions = {}) {
    this.nComponents = options.nComponents;
    this.alpha = options.alpha ?? 1;
    this.maxIter = options.maxIter ?? 1000;
    this.tolerance = options.tolerance ?? 1e-8;
    this.randomState = options.randomState;
    this.transformAlpha = options.transformAlpha ?? this.alpha;
    if (this.nComponents !== undefined && (!Number.isInteger(this.nComponents) || this.nComponents < 1)) {
      throw new Error(`nComponents must be an integer >= 1 when provided. Got ${this.nComponents}.`);
    }
    if (!Number.isFinite(this.alpha) || this.alpha < 0) {
      throw new Error(`alpha must be finite and >= 0. Got ${this.alpha}.`);
    }
  }

  fit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    // Practical approximation: initialize dictionary from principal axes.
    const pca = new PCA({
      nComponents: this.nComponents,
      maxIter: this.maxIter,
      tolerance: this.tolerance,
    }).fit(X);
    this.components_ = pca.components_!.map((row) => row.slice());
    this.mean_ = pca.mean_!.slice();
    this.nFeaturesIn_ = X[0].length;
    this.nComponents_ = this.components_.length;
    this.error_ = 0;
    this.fitted = true;
    return this;
  }

  transform(X: Matrix): Matrix {
    this.assertFitted();
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }
    const out: Matrix = new Array(X.length);
    for (let i = 0; i < X.length; i += 1) {
      const centered = new Array<number>(this.nFeaturesIn_!);
      for (let j = 0; j < this.nFeaturesIn_!; j += 1) {
        centered[j] = X[i][j] - this.mean_![j];
      }
      const row = new Array<number>(this.components_!.length);
      for (let c = 0; c < this.components_!.length; c += 1) {
        row[c] = softThreshold(dot(centered, this.components_![c]), this.transformAlpha);
      }
      out[i] = row;
    }
    return out;
  }

  fitTransform(X: Matrix): Matrix {
    return this.fit(X).transform(X);
  }

  inverseTransform(code: Matrix): Matrix {
    this.assertFitted();
    assertNonEmptyMatrix(code);
    assertConsistentRowSize(code);
    assertFiniteMatrix(code);
    if (code[0].length !== this.components_!.length) {
      throw new Error(
        `Code width mismatch. Expected ${this.components_!.length}, got ${code[0].length}.`,
      );
    }
    const out: Matrix = new Array(code.length);
    for (let i = 0; i < code.length; i += 1) {
      const row = this.mean_!.slice();
      for (let c = 0; c < this.components_!.length; c += 1) {
        for (let j = 0; j < row.length; j += 1) {
          row[j] += code[i][c] * this.components_![c][j];
        }
      }
      out[i] = row;
    }
    return out;
  }

  private assertFitted(): void {
    if (!this.fitted || !this.components_ || !this.mean_ || this.nFeaturesIn_ === null) {
      throw new Error("DictionaryLearning has not been fitted.");
    }
  }
}
