import type { Matrix, Vector } from "../types";
import { dot } from "../utils/linalg";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";
import { PCA } from "./PCA";

export interface SparsePCAOptions {
  nComponents?: number;
  alpha?: number;
  maxIter?: number;
  tolerance?: number;
  randomState?: number;
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

function normalizeRow(row: Vector): Vector {
  let normSq = 0;
  for (let i = 0; i < row.length; i += 1) {
    normSq += row[i] * row[i];
  }
  const norm = Math.sqrt(normSq);
  if (norm <= 1e-12) {
    return row.slice();
  }
  return row.map((value) => value / norm);
}

export class SparsePCA {
  components_: Matrix | null = null;
  mean_: Vector | null = null;
  nFeaturesIn_: number | null = null;
  nComponents_: number | null = null;

  private nComponents?: number;
  private alpha: number;
  private maxIter: number;
  private tolerance: number;
  private randomState?: number;
  private fitted = false;

  constructor(options: SparsePCAOptions = {}) {
    this.nComponents = options.nComponents;
    this.alpha = options.alpha ?? 1;
    this.maxIter = options.maxIter ?? 1000;
    this.tolerance = options.tolerance ?? 1e-8;
    this.randomState = options.randomState;
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

    const pca = new PCA({
      nComponents: this.nComponents,
      maxIter: this.maxIter,
      tolerance: this.tolerance,
    }).fit(X);
    const components = pca.components_!.map((row) =>
      normalizeRow(row.map((value) => softThreshold(value, this.alpha))),
    );

    this.components_ = components;
    this.mean_ = pca.mean_!.slice();
    this.nFeaturesIn_ = X[0].length;
    this.nComponents_ = components.length;
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
      for (let k = 0; k < this.components_!.length; k += 1) {
        row[k] = dot(centered, this.components_![k]);
      }
      out[i] = row;
    }
    return out;
  }

  fitTransform(X: Matrix): Matrix {
    return this.fit(X).transform(X);
  }

  inverseTransform(X: Matrix): Matrix {
    this.assertFitted();
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.components_!.length) {
      throw new Error(
        `Component size mismatch. Expected ${this.components_!.length}, got ${X[0].length}.`,
      );
    }
    const out: Matrix = new Array(X.length);
    for (let i = 0; i < X.length; i += 1) {
      const row = this.mean_!.slice();
      for (let k = 0; k < this.components_!.length; k += 1) {
        const coefficient = X[i][k];
        for (let j = 0; j < row.length; j += 1) {
          row[j] += coefficient * this.components_![k][j];
        }
      }
      out[i] = row;
    }
    return out;
  }

  private assertFitted(): void {
    if (!this.fitted || !this.components_ || !this.mean_ || this.nFeaturesIn_ === null) {
      throw new Error("SparsePCA has not been fitted.");
    }
  }
}
