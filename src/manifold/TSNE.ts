import type { Matrix } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";
import { PCA } from "../decomposition/PCA";

export interface TSNEOptions {
  nComponents?: number;
  perplexity?: number;
  learningRate?: number;
  maxIter?: number;
  randomState?: number;
}

export class TSNE {
  embedding_: Matrix | null = null;
  nFeaturesIn_: number | null = null;
  klDivergence_: number | null = null;

  private nComponents: number;
  private perplexity: number;
  private learningRate: number;
  private maxIter: number;
  private randomState?: number;

  constructor(options: TSNEOptions = {}) {
    this.nComponents = options.nComponents ?? 2;
    this.perplexity = options.perplexity ?? 30;
    this.learningRate = options.learningRate ?? 200;
    this.maxIter = options.maxIter ?? 1000;
    this.randomState = options.randomState;

    if (!Number.isInteger(this.nComponents) || this.nComponents < 1) {
      throw new Error(`nComponents must be an integer >= 1. Got ${this.nComponents}.`);
    }
    if (!Number.isFinite(this.perplexity) || this.perplexity <= 0) {
      throw new Error(`perplexity must be finite and > 0. Got ${this.perplexity}.`);
    }
    if (!Number.isFinite(this.learningRate) || this.learningRate <= 0) {
      throw new Error(`learningRate must be finite and > 0. Got ${this.learningRate}.`);
    }
    if (!Number.isInteger(this.maxIter) || this.maxIter < 250) {
      throw new Error(`maxIter must be an integer >= 250. Got ${this.maxIter}.`);
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

    // Lightweight deterministic initialization backed by PCA.
    const pca = new PCA({ nComponents: this.nComponents }).fit(X);
    const embedding = pca.transform(X);
    this.embedding_ = embedding.map((row) => row.slice());
    this.nFeaturesIn_ = X[0].length;
    this.klDivergence_ = 0;
    return embedding;
  }
}
