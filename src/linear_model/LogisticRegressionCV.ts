import type { ClassificationModel, Matrix, Vector } from "../types";
import { accuracyScore } from "../metrics/classification";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  validateClassificationInputs,
} from "../utils/validation";
import { crossValScore } from "../model_selection/crossValScore";
import { LogisticRegression } from "./LogisticRegression";

export interface LogisticRegressionCVOptions {
  Cs?: number[];
  cv?: number;
  fitIntercept?: boolean;
  solver?: "gd" | "lbfgs";
  learningRate?: number;
  maxIter?: number;
  tolerance?: number;
  lbfgsMemory?: number;
}

export class LogisticRegressionCV implements ClassificationModel {
  coef_: Vector | Matrix = [];
  intercept_: number | Vector = 0;
  classes_: Vector = [];
  C_ = 1;
  Cs_: Vector = [];
  scores_: Vector = [];

  private Cs: Vector;
  private cv: number;
  private fitIntercept: boolean;
  private solver: "gd" | "lbfgs";
  private learningRate: number;
  private maxIter: number;
  private tolerance: number;
  private lbfgsMemory: number;
  private bestEstimator_: LogisticRegression | null = null;
  private fitted = false;

  constructor(options: LogisticRegressionCVOptions = {}) {
    this.Cs = options.Cs ?? [0.01, 0.1, 1, 10, 100];
    this.cv = options.cv ?? 5;
    this.fitIntercept = options.fitIntercept ?? true;
    this.solver = options.solver ?? "gd";
    this.learningRate = options.learningRate ?? 0.1;
    this.maxIter = options.maxIter ?? 20_000;
    this.tolerance = options.tolerance ?? 1e-8;
    this.lbfgsMemory = options.lbfgsMemory ?? 7;
    this.validateOptions();
  }

  fit(X: Matrix, y: Vector, sampleWeight?: Vector): this {
    validateClassificationInputs(X, y);

    this.Cs_ = this.Cs.slice();
    this.scores_ = new Array<number>(this.Cs.length).fill(0);

    let bestScore = Number.NEGATIVE_INFINITY;
    let bestC = this.Cs[0];

    for (let i = 0; i < this.Cs.length; i += 1) {
      const C = this.Cs[i];
      const scores = crossValScore(
        () =>
          new LogisticRegression({
            fitIntercept: this.fitIntercept,
            solver: this.solver,
            learningRate: this.learningRate,
            maxIter: this.maxIter,
            tolerance: this.tolerance,
            l2: 1 / C,
            lbfgsMemory: this.lbfgsMemory,
          }),
        X,
        y,
        { cv: this.cv, sampleWeight },
      );

      let meanScore = 0;
      for (let j = 0; j < scores.length; j += 1) {
        meanScore += scores[j];
      }
      meanScore /= scores.length;
      this.scores_[i] = meanScore;

      if (meanScore > bestScore) {
        bestScore = meanScore;
        bestC = C;
      }
    }

    this.C_ = bestC;
    this.bestEstimator_ = new LogisticRegression({
      fitIntercept: this.fitIntercept,
      solver: this.solver,
      learningRate: this.learningRate,
      maxIter: this.maxIter,
      tolerance: this.tolerance,
      l2: 1 / this.C_,
      lbfgsMemory: this.lbfgsMemory,
    }).fit(X, y, sampleWeight);

    this.coef_ = Array.isArray(this.bestEstimator_.coef_[0])
      ? (this.bestEstimator_.coef_ as Matrix).map((row) => row.slice())
      : (this.bestEstimator_.coef_ as Vector).slice();
    this.intercept_ = Array.isArray(this.bestEstimator_.intercept_)
      ? (this.bestEstimator_.intercept_ as Vector).slice()
      : (this.bestEstimator_.intercept_ as number);
    this.classes_ = this.bestEstimator_.classes_.slice();

    this.fitted = true;
    return this;
  }

  predictProba(X: Matrix): Matrix {
    this.assertFitted();
    return this.bestEstimator_!.predictProba(X);
  }

  predict(X: Matrix): Vector {
    this.assertFitted();
    return this.bestEstimator_!.predict(X);
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return accuracyScore(y, this.predict(X));
  }

  private assertFitted(): void {
    if (!this.fitted || !this.bestEstimator_) {
      throw new Error("LogisticRegressionCV has not been fitted.");
    }
  }

  private validateOptions(): void {
    if (!Array.isArray(this.Cs) || this.Cs.length === 0) {
      throw new Error("Cs must be a non-empty numeric array.");
    }
    for (let i = 0; i < this.Cs.length; i += 1) {
      if (!Number.isFinite(this.Cs[i]) || this.Cs[i] <= 0) {
        throw new Error(`Cs must contain finite values > 0. Got ${this.Cs[i]}.`);
      }
    }
    if (!Number.isInteger(this.cv) || this.cv < 2) {
      throw new Error(`cv must be an integer >= 2. Got ${this.cv}.`);
    }
  }
}