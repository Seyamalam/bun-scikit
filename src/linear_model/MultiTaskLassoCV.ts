import type { Matrix } from "../types";
import { meanSquaredError, r2Score } from "../metrics/regression";
import { KFold } from "../model_selection/KFold";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";
import { MultiTaskLasso } from "./MultiTaskLasso";

export interface MultiTaskLassoCVOptions {
  alphas?: number[];
  cv?: number;
  fitIntercept?: boolean;
  maxIter?: number;
  tolerance?: number;
  randomState?: number;
}

function subsetRows(X: Matrix, indices: number[]): Matrix {
  return indices.map((idx) => X[idx]);
}

export class MultiTaskLassoCV {
  alpha_ = 1;
  coef_: Matrix = [];
  intercept_: number[] = [];
  msePath_: number[] = [];

  private alphas: number[];
  private cv: number;
  private fitIntercept: boolean;
  private maxIter: number;
  private tolerance: number;
  private randomState: number;
  private bestEstimator_: MultiTaskLasso | null = null;
  private fitted = false;

  constructor(options: MultiTaskLassoCVOptions = {}) {
    this.alphas = options.alphas ?? [1e-4, 1e-3, 1e-2, 1e-1, 1, 10];
    this.cv = options.cv ?? 5;
    this.fitIntercept = options.fitIntercept ?? true;
    this.maxIter = options.maxIter ?? 1000;
    this.tolerance = options.tolerance ?? 1e-4;
    this.randomState = options.randomState ?? 42;
  }

  fit(X: Matrix, Y: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    assertNonEmptyMatrix(Y, "Y");
    assertConsistentRowSize(Y, "Y");
    assertFiniteMatrix(Y, "Y");
    if (X.length !== Y.length) {
      throw new Error(`X and Y row counts must match. Got ${X.length} and ${Y.length}.`);
    }

    const splitter = new KFold({ nSplits: this.cv, shuffle: true, randomState: this.randomState });
    const folds = splitter.split(X, Y);

    this.msePath_ = new Array<number>(this.alphas.length).fill(0);
    let bestAlpha = this.alphas[0];
    let bestMse = Number.POSITIVE_INFINITY;

    for (let a = 0; a < this.alphas.length; a += 1) {
      const alpha = this.alphas[a];
      let mseTotal = 0;
      for (let f = 0; f < folds.length; f += 1) {
        const fold = folds[f];
        const XTrain = subsetRows(X, fold.trainIndices);
        const YTrain = subsetRows(Y, fold.trainIndices);
        const XTest = subsetRows(X, fold.testIndices);
        const YTest = subsetRows(Y, fold.testIndices);

        const estimator = new MultiTaskLasso({
          alpha,
          fitIntercept: this.fitIntercept,
          maxIter: this.maxIter,
          tolerance: this.tolerance,
        }).fit(XTrain, YTrain);

        mseTotal += meanSquaredError(YTest, estimator.predict(XTest)) as number;
      }
      const mseMean = mseTotal / folds.length;
      this.msePath_[a] = mseMean;
      if (mseMean < bestMse) {
        bestMse = mseMean;
        bestAlpha = alpha;
      }
    }

    this.alpha_ = bestAlpha;
    this.bestEstimator_ = new MultiTaskLasso({
      alpha: this.alpha_,
      fitIntercept: this.fitIntercept,
      maxIter: this.maxIter,
      tolerance: this.tolerance,
    }).fit(X, Y);

    this.coef_ = this.bestEstimator_.coef_.map((row) => row.slice());
    this.intercept_ = this.bestEstimator_.intercept_.slice();
    this.fitted = true;
    return this;
  }

  predict(X: Matrix): Matrix {
    this.assertFitted();
    return this.bestEstimator_!.predict(X);
  }

  score(X: Matrix, Y: Matrix): number {
    return r2Score(Y, this.predict(X)) as number;
  }

  private assertFitted(): void {
    if (!this.fitted || !this.bestEstimator_) {
      throw new Error("MultiTaskLassoCV has not been fitted.");
    }
  }
}