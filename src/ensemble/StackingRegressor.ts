import type { Matrix, Vector } from "../types";
import { r2Score } from "../metrics/regression";
import { KFold } from "../model_selection/KFold";
import { assertFiniteVector, validateRegressionInputs } from "../utils/validation";
import {
  fitWithSampleWeight,
  subsetSampleWeight,
  type FitSampleWeightRequest,
} from "../utils/fitWithSampleWeight";

type RegressorLike = {
  fit(X: Matrix, y: Vector, sampleWeight?: Vector): unknown;
  predict(X: Matrix): Vector;
  score?(X: Matrix, y: Vector): number;
};

export type StackingRegressorEstimatorSpec = [
  name: string,
  estimatorFactory: (() => RegressorLike) | RegressorLike,
];

export interface StackingRegressorOptions {
  cv?: number;
  passthrough?: boolean;
  randomState?: number;
}

function subsetMatrix(X: Matrix, indices: number[]): Matrix {
  const out = new Array<Matrix[number]>(indices.length);
  for (let i = 0; i < indices.length; i += 1) {
    out[i] = X[indices[i]];
  }
  return out;
}

function subsetVector(y: Vector, indices: number[]): Vector {
  const out = new Array<number>(indices.length);
  for (let i = 0; i < indices.length; i += 1) {
    out[i] = y[indices[i]];
  }
  return out;
}

function resolveEstimator(input: (() => RegressorLike) | RegressorLike): RegressorLike {
  if (typeof input === "function") {
    return input();
  }
  return input;
}

export class StackingRegressor {
  estimators_: Array<[string, RegressorLike]> = [];
  finalEstimator_: RegressorLike | null = null;
  sampleWeightRequest_ = true;

  private readonly estimatorSpecs: StackingRegressorEstimatorSpec[];
  private readonly finalEstimatorFactory: (() => RegressorLike) | RegressorLike;
  private readonly cv: number;
  private readonly passthrough: boolean;
  private readonly randomState: number;
  private isFitted = false;

  constructor(
    estimators: StackingRegressorEstimatorSpec[],
    finalEstimator: (() => RegressorLike) | RegressorLike,
    options: StackingRegressorOptions = {},
  ) {
    if (!Array.isArray(estimators) || estimators.length === 0) {
      throw new Error("StackingRegressor requires at least one base estimator.");
    }
    this.estimatorSpecs = estimators;
    this.finalEstimatorFactory = finalEstimator;
    this.cv = options.cv ?? 5;
    this.passthrough = options.passthrough ?? false;
    this.randomState = options.randomState ?? 42;

    if (!Number.isInteger(this.cv) || this.cv < 2) {
      throw new Error(`cv must be an integer >= 2. Got ${this.cv}.`);
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
    if (this.cv > X.length) {
      throw new Error(`cv (${this.cv}) cannot exceed sample count (${X.length}).`);
    }

    const seen = new Set<string>();
    for (let i = 0; i < this.estimatorSpecs.length; i += 1) {
      const name = this.estimatorSpecs[i][0];
      if (seen.has(name)) {
        throw new Error(`Estimator names must be unique. Duplicate '${name}'.`);
      }
      seen.add(name);
    }

    const splitter = new KFold({
      nSplits: this.cv,
      shuffle: true,
      randomState: this.randomState,
    });
    const folds = splitter.split(X, y);
    const metaFeatures = Array.from({ length: X.length }, () =>
      new Array<number>(this.estimatorSpecs.length).fill(0),
    );

    for (let estimatorIndex = 0; estimatorIndex < this.estimatorSpecs.length; estimatorIndex += 1) {
      const [, estimatorFactory] = this.estimatorSpecs[estimatorIndex];
      for (let foldIndex = 0; foldIndex < folds.length; foldIndex += 1) {
        const fold = folds[foldIndex];
        const estimator = resolveEstimator(estimatorFactory);
        fitWithSampleWeight(
          estimator,
          subsetMatrix(X, fold.trainIndices),
          subsetVector(y, fold.trainIndices),
          subsetSampleWeight(routedSampleWeight, fold.trainIndices),
        );
        const predictions = estimator.predict(subsetMatrix(X, fold.testIndices));
        for (let i = 0; i < fold.testIndices.length; i += 1) {
          metaFeatures[fold.testIndices[i]][estimatorIndex] = predictions[i];
        }
      }
    }

    this.estimators_ = this.estimatorSpecs.map(([name, estimatorFactory]) => {
      const estimator = resolveEstimator(estimatorFactory);
      fitWithSampleWeight(estimator, X, y, routedSampleWeight);
      return [name, estimator];
    });

    const finalEstimator = resolveEstimator(this.finalEstimatorFactory);
    fitWithSampleWeight(finalEstimator, this.buildFinalFeatures(metaFeatures, X), y, routedSampleWeight);
    this.finalEstimator_ = finalEstimator;
    this.isFitted = true;
    return this;
  }

  predict(X: Matrix): Vector {
    this.assertFitted();

    const metaFeatures: Matrix = Array.from({ length: X.length }, () =>
      new Array<number>(this.estimators_.length).fill(0),
    );
    for (let estimatorIndex = 0; estimatorIndex < this.estimators_.length; estimatorIndex += 1) {
      const estimator = this.estimators_[estimatorIndex][1];
      const predictions = estimator.predict(X);
      for (let sampleIndex = 0; sampleIndex < X.length; sampleIndex += 1) {
        metaFeatures[sampleIndex][estimatorIndex] = predictions[sampleIndex];
      }
    }

    const finalX = this.buildFinalFeatures(metaFeatures, X);
    return this.finalEstimator_!.predict(finalX);
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

  private buildFinalFeatures(metaFeatures: Matrix, X: Matrix): Matrix {
    if (!this.passthrough) {
      return metaFeatures;
    }
    const out = new Array<Matrix[number]>(metaFeatures.length);
    for (let i = 0; i < metaFeatures.length; i += 1) {
      out[i] = metaFeatures[i].concat(X[i]);
    }
    return out;
  }

  private assertFitted(): void {
    if (!this.isFitted || this.estimators_.length === 0 || !this.finalEstimator_) {
      throw new Error("StackingRegressor has not been fitted.");
    }
  }
}

