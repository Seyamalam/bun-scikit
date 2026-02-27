import type { Matrix, Vector } from "../types";
import { accuracyScore } from "../metrics/classification";
import { StratifiedKFold } from "../model_selection/StratifiedKFold";
import {
  argmax,
  buildLabelIndex,
  uniqueSortedLabels,
} from "../utils/classification";
import { assertFiniteVector, validateClassificationInputs } from "../utils/validation";

type ClassifierLike = {
  classes_?: Vector | null;
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
  private classToIndex: Map<number, number> = new Map<number, number>();

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

    this.classes_ = uniqueSortedLabels(y);
    this.classToIndex = buildLabelIndex(this.classes_);
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
    const metaBlocks: Matrix[] = new Array(this.estimatorSpecs.length);

    for (let estimatorIndex = 0; estimatorIndex < this.estimatorSpecs.length; estimatorIndex += 1) {
      const [, estimatorFactory] = this.estimatorSpecs[estimatorIndex];
      let blockWidth = -1;
      let block: Matrix = [];

      for (let foldIndex = 0; foldIndex < folds.length; foldIndex += 1) {
        const fold = folds[foldIndex];
        const estimator = resolveEstimator(estimatorFactory);
        estimator.fit(subsetMatrix(X, fold.trainIndices), subsetVector(y, fold.trainIndices));
        const stackBlock = this.stackBlock(estimator, subsetMatrix(X, fold.testIndices));
        if (blockWidth === -1) {
          blockWidth = stackBlock[0].length;
          block = Array.from({ length: X.length }, () => new Array<number>(blockWidth).fill(0));
        }
        for (let i = 0; i < fold.testIndices.length; i += 1) {
          block[fold.testIndices[i]] = stackBlock[i];
        }
      }
      metaBlocks[estimatorIndex] = block;
    }

    const metaFeatures = this.hstack(metaBlocks);
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
    const metaBlocks = this.estimators_.map(([, estimator]) => this.stackBlock(estimator, X));
    const finalX = this.buildFinalFeatures(this.hstack(metaBlocks), X);

    if (typeof this.finalEstimator_!.predictProba === "function") {
      const proba = this.finalEstimator_!.predictProba(finalX);
      if (proba.length > 0 && proba[0].length === this.classes_.length) {
        return proba;
      }
    }

    const labels = this.finalEstimator_!.predict(finalX);
    return labels.map((label) => {
      const row = new Array<number>(this.classes_.length).fill(0);
      const classIndex = this.classToIndex.get(label);
      if (classIndex !== undefined) {
        row[classIndex] = 1;
      }
      return row;
    });
  }

  predict(X: Matrix): Vector {
    this.assertFitted();
    const proba = this.predictProba(X);
    return proba.map((row) => this.classes_[argmax(row)]);
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return accuracyScore(y, this.predict(X));
  }

  private stackBlock(estimator: ClassifierLike, X: Matrix): Matrix {
    if (this.stackMethod === "predict") {
      return estimator.predict(X).map((value) => [value]);
    }

    if (this.stackMethod === "predictProba") {
      if (typeof estimator.predictProba !== "function") {
        throw new Error("stackMethod='predictProba' requires base estimators with predictProba().");
      }
      return this.alignEstimatorProba(estimator, estimator.predictProba(X));
    }

    if (typeof estimator.predictProba === "function") {
      return this.alignEstimatorProba(estimator, estimator.predictProba(X));
    }
    return estimator.predict(X).map((value) => [value]);
  }

  private alignEstimatorProba(estimator: ClassifierLike, proba: Matrix): Matrix {
    const estimatorClasses =
      estimator.classes_ && estimator.classes_.length > 0 ? estimator.classes_ : this.classes_;
    const estimatorClassToIndex = buildLabelIndex(estimatorClasses);
    const out: Matrix = new Array(proba.length);
    for (let i = 0; i < proba.length; i += 1) {
      const row = new Array<number>(this.classes_.length).fill(0);
      for (let classIndex = 0; classIndex < this.classes_.length; classIndex += 1) {
        const sourceIndex = estimatorClassToIndex.get(this.classes_[classIndex]);
        if (sourceIndex !== undefined && sourceIndex < proba[i].length) {
          row[classIndex] = proba[i][sourceIndex];
        }
      }
      out[i] = row;
    }
    return out;
  }

  private hstack(blocks: Matrix[]): Matrix {
    const rows = blocks[0].length;
    const out: Matrix = new Array(rows);
    for (let i = 0; i < rows; i += 1) {
      const row: number[] = [];
      for (let b = 0; b < blocks.length; b += 1) {
        row.push(...blocks[b][i]);
      }
      out[i] = row;
    }
    return out;
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
