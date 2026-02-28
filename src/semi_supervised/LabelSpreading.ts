import type { ClassificationModel, Matrix, Vector } from "../types";
import { accuracyScore } from "../metrics/classification";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
} from "../utils/validation";
import {
  buildTransitionMatrix,
  initializeState,
  maxAbsDiff,
  multiply,
  normalizeRowsInPlace,
  queryWeights,
  transductionFromDistributions,
  type SemiSupervisedKernel,
  validateSemiSupervisedInputs,
} from "./shared";

export interface LabelSpreadingOptions {
  kernel?: SemiSupervisedKernel;
  gamma?: number;
  nNeighbors?: number;
  alpha?: number;
  maxIter?: number;
  tolerance?: number;
}

export class LabelSpreading implements ClassificationModel {
  classes_: Vector = [0, 1];
  labelDistributions_: Matrix | null = null;
  transduction_: Vector | null = null;
  nIter_ = 0;
  nFeaturesIn_: number | null = null;

  private kernel: SemiSupervisedKernel;
  private gamma: number;
  private nNeighbors: number;
  private alpha: number;
  private maxIter: number;
  private tolerance: number;
  private XTrain: Matrix | null = null;
  private fitted = false;

  constructor(options: LabelSpreadingOptions = {}) {
    this.kernel = options.kernel ?? "rbf";
    this.gamma = options.gamma ?? 20;
    this.nNeighbors = options.nNeighbors ?? 7;
    this.alpha = options.alpha ?? 0.2;
    this.maxIter = options.maxIter ?? 30;
    this.tolerance = options.tolerance ?? 1e-3;

    if (!(this.kernel === "rbf" || this.kernel === "knn")) {
      throw new Error(`kernel must be 'rbf' or 'knn'. Got ${this.kernel}.`);
    }
    if (!Number.isFinite(this.gamma) || this.gamma <= 0) {
      throw new Error(`gamma must be finite and > 0. Got ${this.gamma}.`);
    }
    if (!Number.isInteger(this.nNeighbors) || this.nNeighbors < 1) {
      throw new Error(`nNeighbors must be an integer >= 1. Got ${this.nNeighbors}.`);
    }
    if (!Number.isFinite(this.alpha) || this.alpha <= 0 || this.alpha >= 1) {
      throw new Error(`alpha must be finite and in (0, 1). Got ${this.alpha}.`);
    }
    if (!Number.isInteger(this.maxIter) || this.maxIter < 1) {
      throw new Error(`maxIter must be an integer >= 1. Got ${this.maxIter}.`);
    }
    if (!Number.isFinite(this.tolerance) || this.tolerance < 0) {
      throw new Error(`tolerance must be finite and >= 0. Got ${this.tolerance}.`);
    }
  }

  fit(X: Matrix, y: Vector, sampleWeight?: Vector): this {
    validateSemiSupervisedInputs(X, y);
    const state = initializeState(y);
    const transition = buildTransitionMatrix(X, {
      kernel: this.kernel,
      gamma: this.gamma,
      nNeighbors: this.nNeighbors,
    });

    let F = state.Y.map((row) => row.slice());
    let iterations = 0;
    for (let iter = 0; iter < this.maxIter; iter += 1) {
      iterations = iter + 1;
      const propagated = multiply(transition, F);
      const next: Matrix = new Array(propagated.length);
      for (let i = 0; i < propagated.length; i += 1) {
        const row = new Array<number>(propagated[i].length);
        for (let j = 0; j < row.length; j += 1) {
          row[j] = this.alpha * propagated[i][j] + (1 - this.alpha) * state.Y[i][j];
        }
        next[i] = row;
      }
      normalizeRowsInPlace(next);
      const diff = maxAbsDiff(next, F);
      F = next;
      if (diff < this.tolerance) {
        break;
      }
    }

    this.classes_ = state.classes.slice();
    this.labelDistributions_ = F;
    this.transduction_ = transductionFromDistributions(F, this.classes_);
    this.nIter_ = iterations;
    this.nFeaturesIn_ = X[0].length;
    this.XTrain = X.map((row) => row.slice());
    this.fitted = true;
    return this;
  }

  predictProba(X: Matrix): Matrix {
    this.assertFitted();
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }

    const weights = queryWeights(X, this.XTrain!, {
      kernel: this.kernel,
      gamma: this.gamma,
      nNeighbors: this.nNeighbors,
    });
    const out = multiply(weights, this.labelDistributions_!);
    normalizeRowsInPlace(out);
    return out;
  }

  predict(X: Matrix): Vector {
    return this.predictProba(X).map((row) => this.classes_[row.indexOf(Math.max(...row))]);
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return accuracyScore(y, this.predict(X));
  }

  private assertFitted(): void {
    if (
      !this.fitted ||
      !this.XTrain ||
      !this.labelDistributions_ ||
      !this.transduction_ ||
      this.nFeaturesIn_ === null
    ) {
      throw new Error("LabelSpreading has not been fitted.");
    }
  }
}
