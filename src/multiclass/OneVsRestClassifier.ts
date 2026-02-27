import type { Matrix, Vector } from "../types";
import { accuracyScore } from "../metrics/classification";
import { argmax, uniqueSortedLabels } from "../utils/classification";
import { fitWithSampleWeight, type FitSampleWeightRequest } from "../utils/fitWithSampleWeight";
import { assertFiniteVector, validateClassificationInputs } from "../utils/validation";

type ClassifierLike = {
  classes_?: Vector | null;
  fit(X: Matrix, y: Vector, sampleWeight?: Vector): unknown;
  predict(X: Matrix): Vector;
  predictProba?: (X: Matrix) => Matrix;
};

function resolveEstimator(input: (() => ClassifierLike) | ClassifierLike): ClassifierLike {
  if (typeof input === "function") {
    return input();
  }
  return input;
}

function positiveClassProbability(estimator: ClassifierLike, X: Matrix): Vector {
  if (typeof estimator.predictProba === "function") {
    const proba = estimator.predictProba(X);
    let positiveIndex = proba[0].length - 1;
    const classes = estimator.classes_;
    if (classes && classes.length > 0) {
      const idx = classes.indexOf(1);
      if (idx >= 0) {
        positiveIndex = idx;
      }
    }
    return proba.map((row) => row[Math.max(0, Math.min(positiveIndex, row.length - 1))]);
  }
  return estimator.predict(X).map((value) => (value > 0 ? 1 : 0));
}

export interface OneVsRestClassifierOptions {
  normalizeProba?: boolean;
}

export class OneVsRestClassifier {
  classes_: Vector = [0, 1];
  estimators_: ClassifierLike[] = [];
  sampleWeightRequest_ = true;

  private estimatorFactory: (() => ClassifierLike) | ClassifierLike;
  private normalizeProba: boolean;
  private isFitted = false;

  constructor(
    estimatorFactory: (() => ClassifierLike) | ClassifierLike,
    options: OneVsRestClassifierOptions = {},
  ) {
    this.estimatorFactory = estimatorFactory;
    this.normalizeProba = options.normalizeProba ?? true;
  }

  fit(X: Matrix, y: Vector, sampleWeight?: Vector): this {
    validateClassificationInputs(X, y);
    if (sampleWeight) {
      if (sampleWeight.length !== X.length) {
        throw new Error(
          `sampleWeight length must match sample count. Got ${sampleWeight.length} and ${X.length}.`,
        );
      }
      assertFiniteVector(sampleWeight);
    }

    this.classes_ = uniqueSortedLabels(y);
    this.estimators_ = new Array<ClassifierLike>(this.classes_.length);
    const routedSampleWeight = this.sampleWeightRequest_ ? sampleWeight : undefined;

    for (let classIndex = 0; classIndex < this.classes_.length; classIndex += 1) {
      const positiveClass = this.classes_[classIndex];
      const yBinary = y.map((label) => (label === positiveClass ? 1 : 0));
      const estimator = resolveEstimator(this.estimatorFactory);
      fitWithSampleWeight(estimator, X, yBinary, routedSampleWeight);
      this.estimators_[classIndex] = estimator;
    }

    this.isFitted = true;
    return this;
  }

  predictProba(X: Matrix): Matrix {
    this.assertFitted();
    const scores: Matrix = Array.from({ length: X.length }, () =>
      new Array<number>(this.classes_.length).fill(0),
    );

    for (let classIndex = 0; classIndex < this.estimators_.length; classIndex += 1) {
      const probability = positiveClassProbability(this.estimators_[classIndex], X);
      for (let sampleIndex = 0; sampleIndex < probability.length; sampleIndex += 1) {
        scores[sampleIndex][classIndex] = probability[sampleIndex];
      }
    }

    if (!this.normalizeProba) {
      return scores;
    }
    return scores.map((row) => {
      let total = 0;
      for (let i = 0; i < row.length; i += 1) {
        total += row[i];
      }
      if (total <= 0) {
        return row.map(() => 1 / row.length);
      }
      return row.map((value) => value / total);
    });
  }

  predict(X: Matrix): Vector {
    return this.predictProba(X).map((row) => this.classes_[argmax(row)]);
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return accuracyScore(y, this.predict(X));
  }

  setFitRequest(request: FitSampleWeightRequest): this {
    if (typeof request.sampleWeight === "boolean") {
      this.sampleWeightRequest_ = request.sampleWeight;
    }
    return this;
  }

  private assertFitted(): void {
    if (!this.isFitted || this.estimators_.length === 0) {
      throw new Error("OneVsRestClassifier has not been fitted.");
    }
  }
}
