import type { Matrix } from "../types";
import { assertConsistentRowSize, assertFiniteMatrix, assertNonEmptyMatrix } from "../utils/validation";
import { KFold } from "../model_selection/KFold";
import { GraphicalLasso, type GraphicalLassoOptions } from "./GraphicalLasso";

export interface GraphicalLassoCVOptions extends Omit<GraphicalLassoOptions, "alpha"> {
  alphas?: number[] | null;
  cv?: number;
}

function selectRows(X: Matrix, indices: number[]): Matrix {
  const out: Matrix = new Array(indices.length);
  for (let i = 0; i < indices.length; i += 1) {
    out[i] = X[indices[i]].slice();
  }
  return out;
}

export class GraphicalLassoCV {
  alpha_: number | null = null;
  covariance_: Matrix | null = null;
  precision_: Matrix | null = null;
  location_: number[] | null = null;
  nFeaturesIn_: number | null = null;

  private alphas: number[] | null;
  private cv: number;
  private maxIter: number;
  private tolerance: number;
  private assumeCentered: boolean;
  private bestEstimator_: GraphicalLasso | null = null;
  private fitted = false;

  constructor(options: GraphicalLassoCVOptions = {}) {
    this.alphas = options.alphas ?? null;
    this.cv = options.cv ?? 5;
    this.maxIter = options.maxIter ?? 100;
    this.tolerance = options.tolerance ?? 1e-4;
    this.assumeCentered = options.assumeCentered ?? false;

    if (this.alphas !== null) {
      if (!Array.isArray(this.alphas) || this.alphas.length === 0) {
        throw new Error("alphas must be null or a non-empty array.");
      }
      for (let i = 0; i < this.alphas.length; i += 1) {
        const value = this.alphas[i];
        if (!Number.isFinite(value) || value < 0) {
          throw new Error(`alphas must contain finite non-negative values. Got ${value}.`);
        }
      }
    }
    if (!Number.isInteger(this.cv) || this.cv < 2) {
      throw new Error(`cv must be an integer >= 2. Got ${this.cv}.`);
    }
  }

  fit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X.length < this.cv) {
      throw new Error(`cv (${this.cv}) cannot exceed sample count (${X.length}).`);
    }

    const candidateAlphas = this.alphas ?? [0.001, 0.01, 0.1, 0.5, 1];
    const folds = new KFold({ nSplits: this.cv, shuffle: true, randomState: 0 }).split(X);

    let bestAlpha = candidateAlphas[0];
    let bestScore = Number.NEGATIVE_INFINITY;

    for (let a = 0; a < candidateAlphas.length; a += 1) {
      const alpha = candidateAlphas[a];
      let scoreSum = 0;
      let scoreCount = 0;
      for (let f = 0; f < folds.length; f += 1) {
        const train = selectRows(X, folds[f].trainIndices);
        const test = selectRows(X, folds[f].testIndices);
        const model = new GraphicalLasso({
          alpha,
          maxIter: this.maxIter,
          tolerance: this.tolerance,
          assumeCentered: this.assumeCentered,
        }).fit(train);
        scoreSum += model.score(test);
        scoreCount += 1;
      }
      const meanScore = scoreCount === 0 ? Number.NEGATIVE_INFINITY : scoreSum / scoreCount;
      if (meanScore > bestScore) {
        bestScore = meanScore;
        bestAlpha = alpha;
      }
    }

    this.bestEstimator_ = new GraphicalLasso({
      alpha: bestAlpha,
      maxIter: this.maxIter,
      tolerance: this.tolerance,
      assumeCentered: this.assumeCentered,
    }).fit(X);

    this.alpha_ = bestAlpha;
    this.covariance_ = this.bestEstimator_.covariance_;
    this.precision_ = this.bestEstimator_.precision_;
    this.location_ = this.bestEstimator_.location_;
    this.nFeaturesIn_ = X[0].length;
    this.fitted = true;
    return this;
  }

  score(X: Matrix): number {
    this.assertFitted();
    return this.bestEstimator_!.score(X);
  }

  scoreSamples(X: Matrix): number[] {
    this.assertFitted();
    return this.bestEstimator_!.scoreSamples(X);
  }

  mahalanobis(X: Matrix): number[] {
    this.assertFitted();
    return this.bestEstimator_!.mahalanobis(X);
  }

  private assertFitted(): void {
    if (
      !this.fitted ||
      this.bestEstimator_ === null ||
      this.alpha_ === null ||
      this.covariance_ === null ||
      this.precision_ === null ||
      this.location_ === null ||
      this.nFeaturesIn_ === null
    ) {
      throw new Error("GraphicalLassoCV has not been fitted.");
    }
  }
}

