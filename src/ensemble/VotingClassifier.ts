import type { Matrix, Vector } from "../types";
import { accuracyScore } from "../metrics/classification";
import {
  assertFiniteVector,
  validateClassificationInputs,
} from "../utils/validation";

export type VotingStrategy = "hard" | "soft";

type ClassifierLike = {
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

    const probabilities = new Array<number>(X.length).fill(0);
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
      for (let j = 0; j < estProba.length; j += 1) {
        probabilities[j] += weight * estProba[j][1];
      }
      weightTotal += weight;
    }

    for (let i = 0; i < probabilities.length; i += 1) {
      probabilities[i] /= weightTotal;
    }

    return probabilities.map((prob) => [1 - prob, prob]);
  }

  predict(X: Matrix): Vector {
    this.assertFitted();

    if (this.voting === "soft") {
      return this.predictProba(X).map((pair) => (pair[1] >= 0.5 ? 1 : 0));
    }

    const votes = new Array<number>(X.length).fill(0);
    let weightTotal = 0;
    for (let i = 0; i < this.estimators_.length; i += 1) {
      const estimator = this.estimators_[i][1];
      const weight = this.weights?.[i] ?? 1;
      const pred = estimator.predict(X);
      for (let j = 0; j < pred.length; j += 1) {
        if (pred[j] === 1) {
          votes[j] += weight;
        }
      }
      weightTotal += weight;
    }

    return votes.map((weightedPositiveVotes) => (weightedPositiveVotes * 2 >= weightTotal ? 1 : 0));
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
