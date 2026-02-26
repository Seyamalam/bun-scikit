import type { Matrix, Vector } from "../types";
import { accuracyScore } from "../metrics/classification";
import { StratifiedKFold } from "../model_selection/StratifiedKFold";
import { assertFiniteVector, validateClassificationInputs } from "../utils/validation";

type ClassifierLike = {
  fit(X: Matrix, y: Vector): unknown;
  predict(X: Matrix): Vector;
  predictProba?: (X: Matrix) => Matrix;
};

export type StackingMethod = "auto" | "predictProba" | "predict";

export type StackingEstimatorSpec = [
  name: string,
  estimatorFactory: (() => ClassifierLike) | ClassifierLike,
];

export interface StackingClassifierOptions {
  cv?: number;
  passthrough?: boolean;
  stackMethod?: StackingMethod;
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

function resolveEstimator(input: (() => ClassifierLike) | ClassifierLike): ClassifierLike {
  if (typeof input === "function") {
    return input();
  }
  return input;
}

export class StackingClassifier {
  classes_: Vector = [0, 1];
  estimators_: Array<[string, ClassifierLike]> = [];
  finalEstimator_: ClassifierLike | null = null;

  private readonly estimatorSpecs: StackingEstimatorSpec[];
  private readonly finalEstimatorFactory: (() => ClassifierLike) | ClassifierLike;
  private readonly cv: number;
  private readonly passthrough: boolean;
  private readonly stackMethod: StackingMethod;
  private readonly randomState: number;
  private isFitted = false;

  constructor(
    estimators: StackingEstimatorSpec[],
    finalEstimator: (() => ClassifierLike) | ClassifierLike,
    options: StackingClassifierOptions = {},
  ) {
    if (!Array.isArray(estimators) || estimators.length === 0) {
      throw new Error("StackingClassifier requires at least one base estimator.");
    }
    this.estimatorSpecs = estimators;
    this.finalEstimatorFactory = finalEstimator;
    this.cv = options.cv ?? 5;
    this.passthrough = options.passthrough ?? false;
    this.stackMethod = options.stackMethod ?? "auto";
    this.randomState = options.randomState ?? 42;

    if (!Number.isInteger(this.cv) || this.cv < 2) {
      throw new Error(`cv must be an integer >= 2. Got ${this.cv}.`);
    }
  }

  fit(X: Matrix, y: Vector): this {
    validateClassificationInputs(X, y);
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

    const splitter = new StratifiedKFold({
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
        estimator.fit(subsetMatrix(X, fold.trainIndices), subsetVector(y, fold.trainIndices));
        const stackValues = this.stackValues(estimator, subsetMatrix(X, fold.testIndices));
        for (let i = 0; i < fold.testIndices.length; i += 1) {
          metaFeatures[fold.testIndices[i]][estimatorIndex] = stackValues[i];
        }
      }
    }

    this.estimators_ = this.estimatorSpecs.map(([name, estimatorFactory]) => {
      const estimator = resolveEstimator(estimatorFactory);
      estimator.fit(X, y);
      return [name, estimator];
    });

    const finalEstimator = resolveEstimator(this.finalEstimatorFactory);
    finalEstimator.fit(this.buildFinalFeatures(metaFeatures, X), y);
    this.finalEstimator_ = finalEstimator;
    this.isFitted = true;
    return this;
  }

  predictProba(X: Matrix): Matrix {
    this.assertFitted();

    const metaFeatures: Matrix = Array.from({ length: X.length }, () =>
      new Array<number>(this.estimators_.length).fill(0),
    );
    for (let estimatorIndex = 0; estimatorIndex < this.estimators_.length; estimatorIndex += 1) {
      const estimator = this.estimators_[estimatorIndex][1];
      const values = this.stackValues(estimator, X);
      for (let sampleIndex = 0; sampleIndex < X.length; sampleIndex += 1) {
        metaFeatures[sampleIndex][estimatorIndex] = values[sampleIndex];
      }
    }

    const finalX = this.buildFinalFeatures(metaFeatures, X);
    if (typeof this.finalEstimator_!.predictProba === "function") {
      return this.finalEstimator_!.predictProba(finalX);
    }
    const labels = this.finalEstimator_!.predict(finalX);
    return labels.map((label) => [1 - label, label]);
  }

  predict(X: Matrix): Vector {
    this.assertFitted();
    if (typeof this.finalEstimator_!.predict === "function") {
      const metaFeatures: Matrix = Array.from({ length: X.length }, () =>
        new Array<number>(this.estimators_.length).fill(0),
      );
      for (let estimatorIndex = 0; estimatorIndex < this.estimators_.length; estimatorIndex += 1) {
        const estimator = this.estimators_[estimatorIndex][1];
        const values = this.stackValues(estimator, X);
        for (let sampleIndex = 0; sampleIndex < X.length; sampleIndex += 1) {
          metaFeatures[sampleIndex][estimatorIndex] = values[sampleIndex];
        }
      }
      const finalX = this.buildFinalFeatures(metaFeatures, X);
      return this.finalEstimator_!.predict(finalX);
    }
    return this.predictProba(X).map((pair) => (pair[1] >= 0.5 ? 1 : 0));
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return accuracyScore(y, this.predict(X));
  }

  private stackValues(estimator: ClassifierLike, X: Matrix): Vector {
    if (this.stackMethod === "predict") {
      return estimator.predict(X);
    }

    if (this.stackMethod === "predictProba") {
      if (typeof estimator.predictProba !== "function") {
        throw new Error("stackMethod='predictProba' requires base estimators with predictProba().");
      }
      return estimator.predictProba(X).map((pair) => pair[1]);
    }

    if (typeof estimator.predictProba === "function") {
      return estimator.predictProba(X).map((pair) => pair[1]);
    }
    return estimator.predict(X);
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
      throw new Error("StackingClassifier has not been fitted.");
    }
  }
}
