import type { ClassificationModel, Matrix, Vector } from "../types";
import { accuracyScore } from "../metrics/classification";
import { StratifiedKFold } from "../model_selection/StratifiedKFold";
import { assertFiniteVector, validateClassificationInputs } from "../utils/validation";
import { RidgeClassifier } from "./RidgeClassifier";

export interface RidgeClassifierCVOptions {
  alphas?: number[];
  cv?: number;
  fitIntercept?: boolean;
  randomState?: number;
}

function crossValidatedAccuracy(
  alpha: number,
  fitIntercept: boolean,
  X: Matrix,
  y: Vector,
  cv: number,
  randomState?: number,
): number {
  const splitter = new StratifiedKFold({
    nSplits: cv,
    shuffle: true,
    randomState: randomState ?? 42,
  });
  const folds = splitter.split(X, y);
  let total = 0;

  for (let i = 0; i < folds.length; i += 1) {
    const fold = folds[i]!;
    const XTrain = fold.trainIndices.map((index) => X[index]!);
    const yTrain = fold.trainIndices.map((index) => y[index]!);
    const XTest = fold.testIndices.map((index) => X[index]!);
    const yTest = fold.testIndices.map((index) => y[index]!);

    const estimator = new RidgeClassifier({ alpha, fitIntercept }).fit(XTrain, yTrain);
    total += accuracyScore(yTest, estimator.predict(XTest));
  }

  return total / folds.length;
}

export class RidgeClassifierCV implements ClassificationModel {
  alpha_ = 1;
  coef_: Vector | Matrix = [];
  intercept_: number | Vector = 0;
  classes_: Vector = [0, 1];
  cvScores_: Vector = [];

  private readonly alphas: number[];
  private readonly cv: number;
  private readonly fitIntercept: boolean;
  private readonly randomState?: number;
  private fitted = false;
  private model: RidgeClassifier | null = null;

  constructor(options: RidgeClassifierCVOptions = {}) {
    this.alphas = options.alphas ?? [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100];
    this.cv = options.cv ?? 5;
    this.fitIntercept = options.fitIntercept ?? true;
    this.randomState = options.randomState;

    if (!Number.isInteger(this.cv) || this.cv < 2) {
      throw new Error(`cv must be an integer >= 2. Got ${this.cv}.`);
    }
    if (!Array.isArray(this.alphas) || this.alphas.length === 0) {
      throw new Error("alphas must be a non-empty numeric array.");
    }
    for (let i = 0; i < this.alphas.length; i += 1) {
      if (!Number.isFinite(this.alphas[i]!) || this.alphas[i]! < 0) {
        throw new Error(`alphas must contain finite values >= 0. Got ${this.alphas[i]}.`);
      }
    }
  }

  fit(X: Matrix, y: Vector, sampleWeight?: Vector): this {
    validateClassificationInputs(X, y);

    let bestAlpha = this.alphas[0]!;
    let bestScore = Number.NEGATIVE_INFINITY;
    this.cvScores_ = new Array<number>(this.alphas.length).fill(0);

    for (let i = 0; i < this.alphas.length; i += 1) {
      const alpha = this.alphas[i]!;
      const meanAccuracy = crossValidatedAccuracy(
        alpha,
        this.fitIntercept,
        X,
        y,
        this.cv,
        this.randomState,
      );
      this.cvScores_[i] = meanAccuracy;
      if (meanAccuracy > bestScore) {
        bestScore = meanAccuracy;
        bestAlpha = alpha;
      }
    }

    this.model = new RidgeClassifier({
      alpha: bestAlpha,
      fitIntercept: this.fitIntercept,
    }).fit(X, y);
    this.alpha_ = bestAlpha;
    this.coef_ = Array.isArray(this.model.coef_[0])
      ? (this.model.coef_ as Matrix).map((row) => row.slice())
      : (this.model.coef_ as Vector).slice();
    this.intercept_ = Array.isArray(this.model.intercept_)
      ? (this.model.intercept_ as Vector).slice()
      : this.model.intercept_;
    this.classes_ = this.model.classes_.slice();
    this.fitted = true;
    return this;
  }

  decisionFunction(X: Matrix): Vector | Matrix {
    this.assertFitted();
    return this.model!.decisionFunction(X);
  }

  predict(X: Matrix): Vector {
    this.assertFitted();
    return this.model!.predict(X);
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return accuracyScore(y, this.predict(X));
  }

  private assertFitted(): void {
    if (!this.fitted || !this.model) {
      throw new Error("RidgeClassifierCV has not been fitted.");
    }
  }
}
