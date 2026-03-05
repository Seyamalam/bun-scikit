import type { ClassificationModel, Matrix, Vector } from "../types";
import { accuracyScore } from "../metrics/classification";
import {
  argmax,
  uniqueSortedLabels,
} from "../utils/classification";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  validateClassificationInputs,
} from "../utils/validation";
import { Ridge } from "./Ridge";

export interface RidgeClassifierOptions {
  alpha?: number;
  fitIntercept?: boolean;
}

export class RidgeClassifier implements ClassificationModel {
  coef_: Vector | Matrix = [];
  intercept_: number | Vector = 0;
  classes_: Vector = [0, 1];

  private readonly alpha: number;
  private readonly fitIntercept: boolean;
  private fitted = false;
  private featureCount = 0;
  private coefMatrix_: Matrix = [];
  private interceptVector_: Vector = [];

  constructor(options: RidgeClassifierOptions = {}) {
    this.alpha = options.alpha ?? 1;
    this.fitIntercept = options.fitIntercept ?? true;
    if (!Number.isFinite(this.alpha) || this.alpha < 0) {
      throw new Error(`alpha must be finite and >= 0. Got ${this.alpha}.`);
    }
  }

  fit(X: Matrix, y: Vector, sampleWeight?: Vector): this {
    validateClassificationInputs(X, y);

    this.classes_ = uniqueSortedLabels(y);
    if (this.classes_.length < 2) {
      throw new Error("RidgeClassifier requires at least two classes.");
    }

    this.featureCount = X[0].length;
    this.coefMatrix_ = new Array<Matrix[number]>(this.classes_.length);
    this.interceptVector_ = new Array<number>(this.classes_.length).fill(0);

    for (let classIndex = 0; classIndex < this.classes_.length; classIndex += 1) {
      const positiveLabel = this.classes_[classIndex]!;
      const binaryTargets = y.map((label) => (label === positiveLabel ? 1 : -1));
      const model = new Ridge({
        alpha: this.alpha,
        fitIntercept: this.fitIntercept,
      }).fit(X, binaryTargets);
      this.coefMatrix_[classIndex] = model.coef_.slice();
      this.interceptVector_[classIndex] = model.intercept_;
    }

    if (this.classes_.length === 2) {
      this.coef_ = this.coefMatrix_[1]!.slice();
      this.intercept_ = this.interceptVector_[1]!;
    } else {
      this.coef_ = this.coefMatrix_.map((row) => row.slice());
      this.intercept_ = this.interceptVector_.slice();
    }

    this.fitted = true;
    return this;
  }

  decisionFunction(X: Matrix): Vector | Matrix {
    this.assertFitted();
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.featureCount) {
      throw new Error(`Feature size mismatch. Expected ${this.featureCount}, got ${X[0].length}.`);
    }

    const scores = X.map((row) =>
      this.coefMatrix_.map(
        (coef, classIndex) =>
          this.interceptVector_[classIndex]! + coef.reduce((sum, value, index) => sum + value * row[index]!, 0),
      ),
    );

    if (this.classes_.length === 2) {
      return scores.map((row) => row[1]!);
    }
    return scores;
  }

  predict(X: Matrix): Vector {
    const decision = this.decisionFunction(X);
    if (this.classes_.length === 2) {
      return (decision as Vector).map((score) => (score >= 0 ? this.classes_[1]! : this.classes_[0]!));
    }
    return (decision as Matrix).map((row) => this.classes_[argmax(row)]!);
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return accuracyScore(y, this.predict(X));
  }

  private assertFitted(): void {
    if (!this.fitted || this.coefMatrix_.length === 0) {
      throw new Error("RidgeClassifier has not been fitted.");
    }
  }
}
