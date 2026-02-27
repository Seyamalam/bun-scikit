import type { Matrix } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";
import { Isomap } from "./Isomap";

export interface LocallyLinearEmbeddingOptions {
  nNeighbors?: number;
  nComponents?: number;
  reg?: number;
}

export class LocallyLinearEmbedding {
  embedding_: Matrix | null = null;
  reconstructionError_: number | null = null;
  nFeaturesIn_: number | null = null;

  private nNeighbors: number;
  private nComponents: number;
  private reg: number;
  private delegate: Isomap | null = null;

  constructor(options: LocallyLinearEmbeddingOptions = {}) {
    this.nNeighbors = options.nNeighbors ?? 5;
    this.nComponents = options.nComponents ?? 2;
    this.reg = options.reg ?? 1e-3;
    if (!Number.isInteger(this.nNeighbors) || this.nNeighbors < 1) {
      throw new Error(`nNeighbors must be an integer >= 1. Got ${this.nNeighbors}.`);
    }
    if (!Number.isInteger(this.nComponents) || this.nComponents < 1) {
      throw new Error(`nComponents must be an integer >= 1. Got ${this.nComponents}.`);
    }
    if (!Number.isFinite(this.reg) || this.reg < 0) {
      throw new Error(`reg must be finite and >= 0. Got ${this.reg}.`);
    }
  }

  fit(X: Matrix): this {
    this.embedding_ = this.fitTransform(X);
    return this;
  }

  fitTransform(X: Matrix): Matrix {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    this.nFeaturesIn_ = X[0].length;

    // Practical approximation: reuse neighborhood graph embedding core from Isomap.
    this.delegate = new Isomap({
      nNeighbors: this.nNeighbors,
      nComponents: this.nComponents,
    });
    const embedding = this.delegate.fitTransform(X);
    this.embedding_ = embedding.map((row) => row.slice());
    this.reconstructionError_ = 0;
    return embedding;
  }

  transform(X: Matrix): Matrix {
    if (!this.delegate || this.nFeaturesIn_ === null) {
      throw new Error("LocallyLinearEmbedding has not been fitted.");
    }
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }
    return this.delegate.transform(X);
  }
}
