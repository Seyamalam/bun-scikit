import type { Matrix, Vector } from "../types";
import { accuracyScore } from "../metrics/classification";
import {
  assertFiniteVector,
  validateBinaryClassificationInputs,
} from "../utils/validation";
import { HistGradientBoostingRegressor } from "./HistGradientBoostingRegressor";

export interface HistGradientBoostingClassifierOptions {
  maxIter?: number;
  learningRate?: number;
  maxBins?: number;
  minSamplesLeaf?: number;
  l2Regularization?: number;
  randomState?: number;
}

function sigmoid(z: number): number {
  if (z >= 0) {
    const expNeg = Math.exp(-z);
    return 1 / (1 + expNeg);
  }
  const expPos = Math.exp(z);
  return expPos / (1 + expPos);
}

export class HistGradientBoostingClassifier {
  classes_: Vector = [0, 1];
  estimators_: HistGradientBoostingRegressor[] = [];
  baselineLogit_: number | null = null;

  private readonly maxIter: number;
  private readonly learningRate: number;
  private readonly maxBins: number;
  private readonly minSamplesLeaf: number;
  private readonly l2Regularization: number;
  private readonly randomState?: number;
  private isFitted = false;

  constructor(options: HistGradientBoostingClassifierOptions = {}) {
    this.maxIter = options.maxIter ?? 100;
    this.learningRate = options.learningRate ?? 0.1;
    this.maxBins = options.maxBins ?? 255;
    this.minSamplesLeaf = options.minSamplesLeaf ?? 20;
    this.l2Regularization = options.l2Regularization ?? 0;
    this.randomState = options.randomState;
  }

  fit(X: Matrix, y: Vector): this {
    validateBinaryClassificationInputs(X, y);
    this.estimators_ = [];

    let positive = 0;
    for (let i = 0; i < y.length; i += 1) {
      positive += y[i];
    }
    const p = Math.min(1 - 1e-12, Math.max(1e-12, positive / y.length));
    this.baselineLogit_ = Math.log(p / (1 - p));

    const logits = new Array<number>(y.length).fill(this.baselineLogit_);
    for (let iter = 0; iter < this.maxIter; iter += 1) {
      const pseudoResiduals = new Array<number>(y.length);
      for (let i = 0; i < y.length; i += 1) {
        pseudoResiduals[i] = y[i] - sigmoid(logits[i]);
      }

      const regressor = new HistGradientBoostingRegressor({
        maxIter: 1,
        learningRate: 1,
        maxBins: this.maxBins,
        minSamplesLeaf: this.minSamplesLeaf,
        l2Regularization: this.l2Regularization,
        randomState: this.randomState === undefined ? undefined : this.randomState + iter + 1,
      }).fit(X, pseudoResiduals);

      const update = regressor.predict(X);
      for (let i = 0; i < logits.length; i += 1) {
        logits[i] += this.learningRate * update[i];
      }
      this.estimators_.push(regressor);
    }

    this.isFitted = true;
    return this;
  }

  decisionFunction(X: Matrix): Vector {
    this.assertFitted();
    const out = new Array<number>(X.length).fill(this.baselineLogit_!);
    for (let i = 0; i < this.estimators_.length; i += 1) {
      const update = this.estimators_[i].predict(X);
      for (let row = 0; row < out.length; row += 1) {
        out[row] += this.learningRate * update[row];
      }
    }
    return out;
  }

  predictProba(X: Matrix): Matrix {
    return this.decisionFunction(X).map((value) => {
      const p1 = sigmoid(value);
      return [1 - p1, p1];
    });
  }

  predict(X: Matrix): Vector {
    return this.predictProba(X).map((row) => (row[1] >= 0.5 ? 1 : 0));
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return accuracyScore(y, this.predict(X));
  }

  private assertFitted(): void {
    if (!this.isFitted || this.baselineLogit_ === null) {
      throw new Error("HistGradientBoostingClassifier has not been fitted.");
    }
  }
}
