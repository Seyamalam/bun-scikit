import type { Matrix, Vector } from "../types";
import { r2Score } from "../metrics/regression";
import {
  assertFiniteVector,
  validateRegressionInputs,
} from "../utils/validation";
import { fitWithSampleWeight, type FitSampleWeightRequest } from "../utils/fitWithSampleWeight";

type RegressorLike = {
  fit(X: Matrix, y: Vector, sampleWeight?: Vector): unknown;
  predict(X: Matrix): Vector;
  score?(X: Matrix, y: Vector): number;
};

export type VotingRegressorEstimatorSpec = [
  name: string,
  estimatorFactory: (() => RegressorLike) | RegressorLike,
];

export interface VotingRegressorOptions {
  weights?: number[];
}

function resolveEstimator(input: (() => RegressorLike) | RegressorLike): RegressorLike {
  if (typeof input === "function") {
    return input();
  }
  return input;
}

export class VotingRegressor {
  estimators_: Array<[string, RegressorLike]> = [];
  namedEstimators_: Record<string, RegressorLike> = {};
  sampleWeightRequest_ = true;

  private readonly estimatorSpecs: VotingRegressorEstimatorSpec[];
  private readonly weights?: number[];
  private isFitted = false;

  constructor(estimators: VotingRegressorEstimatorSpec[], options: VotingRegressorOptions = {}) {
    if (!Array.isArray(estimators) || estimators.length === 0) {
      throw new Error("VotingRegressor requires at least one estimator.");
    }
    this.estimatorSpecs = estimators;
    this.weights = options.weights;

    if (this.weights && this.weights.length !== estimators.length) {
      throw new Error(
        `weights length must match estimator count (${estimators.length}). Got ${this.weights.length}.`,
      );
    }
  }

  fit(X: Matrix, y: Vector, sampleWeight?: Vector): this {
    validateRegressionInputs(X, y);
    if (sampleWeight) {
      if (sampleWeight.length !== X.length) {
        throw new Error(
          `sampleWeight length must match sample count. Got ${sampleWeight.length} and ${X.length}.`,
        );
      }
      assertFiniteVector(sampleWeight);
    }
    const routedSampleWeight = this.sampleWeightRequest_ ? sampleWeight : undefined;
    const seenNames = new Set<string>();
    const estimators: Array<[string, RegressorLike]> = [];

    for (let i = 0; i < this.estimatorSpecs.length; i += 1) {
      const [name, estimatorOrFactory] = this.estimatorSpecs[i];
      if (seenNames.has(name)) {
        throw new Error(`Estimator names must be unique. Duplicate '${name}'.`);
      }
      seenNames.add(name);
      const estimator = resolveEstimator(estimatorOrFactory);
      fitWithSampleWeight(estimator, X, y, routedSampleWeight);
      estimators.push([name, estimator]);
    }

    this.estimators_ = estimators;
    this.namedEstimators_ = Object.fromEntries(estimators);
    this.isFitted = true;
    return this;
  }

  predict(X: Matrix): Vector {
    this.assertFitted();

    const predictions = new Array<number>(X.length).fill(0);
    let weightTotal = 0;

    for (let i = 0; i < this.estimators_.length; i += 1) {
      const estimator = this.estimators_[i][1];
      const weight = this.weights?.[i] ?? 1;
      const estPred = estimator.predict(X);
      for (let j = 0; j < estPred.length; j += 1) {
        predictions[j] += weight * estPred[j];
      }
      weightTotal += weight;
    }

    for (let i = 0; i < predictions.length; i += 1) {
      predictions[i] /= weightTotal;
    }
    return predictions;
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return r2Score(y, this.predict(X));
  }

  setFitRequest(request: FitSampleWeightRequest): this {
    if (typeof request.sampleWeight === "boolean") {
      this.sampleWeightRequest_ = request.sampleWeight;
    }
    return this;
  }

  private assertFitted(): void {
    if (!this.isFitted || this.estimators_.length === 0) {
      throw new Error("VotingRegressor has not been fitted.");
    }
  }
}

