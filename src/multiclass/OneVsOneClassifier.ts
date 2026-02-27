import type { Matrix, Vector } from "../types";
import { accuracyScore } from "../metrics/classification";
import { argmax, uniqueSortedLabels } from "../utils/classification";
import { fitWithSampleWeight, type FitSampleWeightRequest } from "../utils/fitWithSampleWeight";
import {
  assertFiniteVector,
  validateClassificationInputs,
} from "../utils/validation";

type ClassifierLike = {
  classes_?: Vector | null;
  fit(X: Matrix, y: Vector, sampleWeight?: Vector): unknown;
  predict(X: Matrix): Vector;
  predictProba?: (X: Matrix) => Matrix;
};

interface PairEstimator {
  classA: number;
  classB: number;
  estimator: ClassifierLike;
}

function resolveEstimator(input: (() => ClassifierLike) | ClassifierLike): ClassifierLike {
  if (typeof input === "function") {
    return input();
  }
  return input;
}

function mapPairPrediction(value: number, classA: number, classB: number): number {
  return value > 0 ? classB : classA;
}

function pairPositiveProbability(estimator: ClassifierLike, X: Matrix): Vector {
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

export class OneVsOneClassifier {
  classes_: Vector = [0, 1];
  estimators_: PairEstimator[] = [];
  sampleWeightRequest_ = true;

  private estimatorFactory: (() => ClassifierLike) | ClassifierLike;
  private isFitted = false;
  private labelToIndex: Map<number, number> = new Map<number, number>();

  constructor(estimatorFactory: (() => ClassifierLike) | ClassifierLike) {
    this.estimatorFactory = estimatorFactory;
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
    if (this.classes_.length < 2) {
      throw new Error("OneVsOneClassifier requires at least two classes.");
    }
    this.labelToIndex = new Map<number, number>();
    for (let i = 0; i < this.classes_.length; i += 1) {
      this.labelToIndex.set(this.classes_[i], i);
    }

    const routedSampleWeight = this.sampleWeightRequest_ ? sampleWeight : undefined;
    const pairEstimators: PairEstimator[] = [];
    for (let i = 0; i < this.classes_.length; i += 1) {
      for (let j = i + 1; j < this.classes_.length; j += 1) {
        const classA = this.classes_[i];
        const classB = this.classes_[j];

        const subsetX: Matrix = [];
        const subsetY: Vector = [];
        const subsetWeight: Vector = [];
        for (let rowIndex = 0; rowIndex < y.length; rowIndex += 1) {
          if (y[rowIndex] === classA || y[rowIndex] === classB) {
            subsetX.push(X[rowIndex].slice());
            subsetY.push(y[rowIndex] === classB ? 1 : 0);
            if (routedSampleWeight) {
              subsetWeight.push(routedSampleWeight[rowIndex]);
            }
          }
        }

        const estimator = resolveEstimator(this.estimatorFactory);
        fitWithSampleWeight(
          estimator,
          subsetX,
          subsetY,
          routedSampleWeight ? subsetWeight : undefined,
        );
        pairEstimators.push({ classA, classB, estimator });
      }
    }

    this.estimators_ = pairEstimators;
    this.isFitted = true;
    return this;
  }

  predictProba(X: Matrix): Matrix {
    this.assertFitted();
    const votes: Matrix = Array.from({ length: X.length }, () =>
      new Array<number>(this.classes_.length).fill(0),
    );
    const strengths: Matrix = Array.from({ length: X.length }, () =>
      new Array<number>(this.classes_.length).fill(0),
    );

    for (let estimatorIndex = 0; estimatorIndex < this.estimators_.length; estimatorIndex += 1) {
      const pair = this.estimators_[estimatorIndex];
      const predictions = pair.estimator.predict(X);
      const positiveProb = pairPositiveProbability(pair.estimator, X);

      const classAIndex = this.labelToIndex.get(pair.classA)!;
      const classBIndex = this.labelToIndex.get(pair.classB)!;
      for (let sampleIndex = 0; sampleIndex < predictions.length; sampleIndex += 1) {
        const mapped = mapPairPrediction(predictions[sampleIndex], pair.classA, pair.classB);
        if (mapped === pair.classA) {
          votes[sampleIndex][classAIndex] += 1;
        } else {
          votes[sampleIndex][classBIndex] += 1;
        }
        strengths[sampleIndex][classAIndex] += 1 - positiveProb[sampleIndex];
        strengths[sampleIndex][classBIndex] += positiveProb[sampleIndex];
      }
    }

    return votes.map((row, rowIndex) => {
      let totalVotes = 0;
      for (let i = 0; i < row.length; i += 1) {
        totalVotes += row[i];
      }
      if (totalVotes <= 0) {
        return row.map(() => 1 / row.length);
      }
      // Softly break ties with accumulated pairwise probabilities.
      const combined = new Array<number>(row.length).fill(0);
      for (let i = 0; i < row.length; i += 1) {
        combined[i] = row[i] + strengths[rowIndex][i] * 1e-3;
      }
      let total = 0;
      for (let i = 0; i < combined.length; i += 1) {
        total += combined[i];
      }
      return combined.map((value) => value / total);
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
      throw new Error("OneVsOneClassifier has not been fitted.");
    }
  }
}
