import type { Matrix, Vector } from "../types";
import { accuracyScore } from "../metrics/classification";
import {
  argmax,
  buildLabelIndex,
  uniqueSortedLabels,
} from "../utils/classification";
import { assertFiniteVector, validateClassificationInputs } from "../utils/validation";

export type VotingStrategy = "hard" | "soft";

type ClassifierLike = {
  classes_?: Vector | null;
  fit(X: Matrix, y: Vector): unknown;
  predict(X: Matrix): Vector;
  predictProba?: (X: Matrix) => Matrix;
  score?(X: Matrix, y: Vector): number;
};

export type VotingEstimatorSpec = [
  name: string,
  estimatorFactory: (() => ClassifierLike) | ClassifierLike,
];

export interface VotingClassifierOptions {
  voting?: VotingStrategy;
  weights?: number[];
}

function resolveEstimator(input: (() => ClassifierLike) | ClassifierLike): ClassifierLike {
  if (typeof input === "function") {
    return input();
  }
  return input;
}

export class VotingClassifier {
  classes_: Vector = [0, 1];
  estimators_: Array<[string, ClassifierLike]> = [];
  namedEstimators_: Record<string, ClassifierLike> = {};

  private readonly estimatorSpecs: VotingEstimatorSpec[];
  private readonly voting: VotingStrategy;
  private readonly weights?: number[];
  private isFitted = false;
  private labelToIndex: Map<number, number> = new Map<number, number>();

  constructor(estimators: VotingEstimatorSpec[], options: VotingClassifierOptions = {}) {
    if (!Array.isArray(estimators) || estimators.length === 0) {
      throw new Error("VotingClassifier requires at least one estimator.");
    }
    this.estimatorSpecs = estimators;
    this.voting = options.voting ?? "hard";
    this.weights = options.weights;

    if (this.weights && this.weights.length !== estimators.length) {
      throw new Error(
        `weights length must match estimator count (${estimators.length}). Got ${this.weights.length}.`,
      );
    }
  }

  fit(X: Matrix, y: Vector): this {
    validateClassificationInputs(X, y);
    this.classes_ = uniqueSortedLabels(y);
    this.labelToIndex = buildLabelIndex(this.classes_);

    const seenNames = new Set<string>();
    const estimators: Array<[string, ClassifierLike]> = [];

    for (let i = 0; i < this.estimatorSpecs.length; i += 1) {
      const [name, estimatorOrFactory] = this.estimatorSpecs[i];
      if (seenNames.has(name)) {
        throw new Error(`Estimator names must be unique. Duplicate '${name}'.`);
      }
      seenNames.add(name);
      const estimator = resolveEstimator(estimatorOrFactory);
      estimator.fit(X, y);
      estimators.push([name, estimator]);
    }

    this.estimators_ = estimators;
    this.namedEstimators_ = Object.fromEntries(estimators);
    this.isFitted = true;
    return this;
  }

  predictProba(X: Matrix): Matrix {
    this.assertFitted();

    const probabilities: Matrix = Array.from({ length: X.length }, () =>
      new Array<number>(this.classes_.length).fill(0),
    );
    let weightTotal = 0;

    for (let i = 0; i < this.estimators_.length; i += 1) {
      const estimator = this.estimators_[i][1];
      if (typeof estimator.predictProba !== "function") {
        throw new Error(
          "VotingClassifier predictProba requires all estimators to implement predictProba().",
        );
      }
      const weight = this.weights?.[i] ?? 1;
      const estProba = estimator.predictProba(X);
      const estClasses = estimator.classes_ && estimator.classes_.length > 0
        ? estimator.classes_
        : this.classes_;
      const estLabelToIndex = buildLabelIndex(estClasses);

      for (let sampleIndex = 0; sampleIndex < estProba.length; sampleIndex += 1) {
        for (let classIndex = 0; classIndex < this.classes_.length; classIndex += 1) {
          const label = this.classes_[classIndex];
          const sourceIndex = estLabelToIndex.get(label);
          if (sourceIndex === undefined || sourceIndex >= estProba[sampleIndex].length) {
            continue;
          }
          probabilities[sampleIndex][classIndex] += weight * estProba[sampleIndex][sourceIndex];
        }
      }
      weightTotal += weight;
    }

    for (let i = 0; i < probabilities.length; i += 1) {
      for (let j = 0; j < probabilities[i].length; j += 1) {
        probabilities[i][j] /= weightTotal;
      }
    }

    return probabilities;
  }

  predict(X: Matrix): Vector {
    this.assertFitted();

    if (this.voting === "soft") {
      return this.predictProba(X).map((row) => this.classes_[argmax(row)]);
    }

    const voteTotals: Matrix = Array.from({ length: X.length }, () =>
      new Array<number>(this.classes_.length).fill(0),
    );

    for (let i = 0; i < this.estimators_.length; i += 1) {
      const estimator = this.estimators_[i][1];
      const weight = this.weights?.[i] ?? 1;
      const pred = estimator.predict(X);
      for (let sampleIndex = 0; sampleIndex < pred.length; sampleIndex += 1) {
        const classIndex = this.labelToIndex.get(pred[sampleIndex]);
        if (classIndex !== undefined) {
          voteTotals[sampleIndex][classIndex] += weight;
        }
      }
    }

    return voteTotals.map((row) => this.classes_[argmax(row)]);
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return accuracyScore(y, this.predict(X));
  }

  private assertFitted(): void {
    if (!this.isFitted || this.estimators_.length === 0) {
      throw new Error("VotingClassifier has not been fitted.");
    }
  }
}
