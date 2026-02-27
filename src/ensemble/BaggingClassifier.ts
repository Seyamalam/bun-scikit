import type { Matrix, Vector } from "../types";
import { accuracyScore } from "../metrics/classification";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  assertNonEmptyMatrix,
  validateClassificationInputs,
} from "../utils/validation";

type ClassifierLike = {
  fit(X: Matrix, y: Vector): unknown;
  predict(X: Matrix): Vector;
  predictProba?: (X: Matrix) => Matrix;
};

export interface BaggingClassifierOptions {
  nEstimators?: number;
  maxSamples?: number;
  maxFeatures?: number;
  bootstrap?: boolean;
  bootstrapFeatures?: boolean;
  randomState?: number;
}

class Mulberry32 {
  private state: number;

  constructor(seed: number) {
    this.state = seed >>> 0;
  }

  next(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let t = this.state ^ (this.state >>> 15);
    t = Math.imul(t, this.state | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
}

function resolveEstimator(input: (() => ClassifierLike) | ClassifierLike): ClassifierLike {
  if (typeof input === "function") {
    return input();
  }
  return input;
}

function resolveSubsetSize(
  value: number | undefined,
  total: number,
  label: string,
): number {
  if (value === undefined) {
    return total;
  }
  if (value > 0 && value <= 1) {
    return Math.max(1, Math.floor(total * value));
  }
  if (Number.isInteger(value) && value >= 1 && value <= total) {
    return value;
  }
  throw new Error(`${label} must be in (0, 1] or integer in [1, ${total}]. Got ${value}.`);
}

function subsetRowsAndColumns(
  X: Matrix,
  rowIndices: number[],
  featureIndices: number[],
): Matrix {
  const out: Matrix = new Array(rowIndices.length);
  for (let i = 0; i < rowIndices.length; i += 1) {
    const row = X[rowIndices[i]];
    const projected = new Array<number>(featureIndices.length);
    for (let j = 0; j < featureIndices.length; j += 1) {
      projected[j] = row[featureIndices[j]];
    }
    out[i] = projected;
  }
  return out;
}

function subsetColumns(X: Matrix, featureIndices: number[]): Matrix {
  const out: Matrix = new Array(X.length);
  for (let i = 0; i < X.length; i += 1) {
    const row = new Array<number>(featureIndices.length);
    for (let j = 0; j < featureIndices.length; j += 1) {
      row[j] = X[i][featureIndices[j]];
    }
    out[i] = row;
  }
  return out;
}

function sampleWithReplacement(size: number, maxExclusive: number, random: () => number): number[] {
  const out = new Array<number>(size);
  for (let i = 0; i < size; i += 1) {
    out[i] = Math.floor(random() * maxExclusive);
  }
  return out;
}

function sampleWithoutReplacement(
  size: number,
  maxExclusive: number,
  random: () => number,
): number[] {
  const pool = Array.from({ length: maxExclusive }, (_, idx) => idx);
  for (let i = pool.length - 1; i > 0; i -= 1) {
    const j = Math.floor(random() * (i + 1));
    const tmp = pool[i];
    pool[i] = pool[j];
    pool[j] = tmp;
  }
  return pool.slice(0, size);
}

function hasBothClasses(y: Vector, indices: number[]): boolean {
  let hasZero = false;
  let hasOne = false;
  for (let i = 0; i < indices.length; i += 1) {
    const label = y[indices[i]];
    if (label === 0) {
      hasZero = true;
    } else if (label === 1) {
      hasOne = true;
    }
    if (hasZero && hasOne) {
      return true;
    }
  }
  return false;
}

export class BaggingClassifier {
  classes_: Vector = [0, 1];
  estimators_: ClassifierLike[] = [];
  estimatorsFeatures_: number[][] = [];

  private readonly estimatorFactory: (() => ClassifierLike) | ClassifierLike;
  private readonly nEstimators: number;
  private readonly maxSamples?: number;
  private readonly maxFeatures?: number;
  private readonly bootstrap: boolean;
  private readonly bootstrapFeatures: boolean;
  private readonly randomState?: number;
  private nFeaturesIn_: number | null = null;

  constructor(
    estimatorFactory: (() => ClassifierLike) | ClassifierLike,
    options: BaggingClassifierOptions = {},
  ) {
    this.estimatorFactory = estimatorFactory;
    this.nEstimators = options.nEstimators ?? 10;
    this.maxSamples = options.maxSamples;
    this.maxFeatures = options.maxFeatures;
    this.bootstrap = options.bootstrap ?? true;
    this.bootstrapFeatures = options.bootstrapFeatures ?? false;
    this.randomState = options.randomState;

    if (!Number.isInteger(this.nEstimators) || this.nEstimators < 1) {
      throw new Error(`nEstimators must be an integer >= 1. Got ${this.nEstimators}.`);
    }
  }

  fit(X: Matrix, y: Vector): this {
    validateClassificationInputs(X, y);

    const nSamples = X.length;
    const nFeatures = X[0].length;
    const sampleCount = resolveSubsetSize(this.maxSamples, nSamples, "maxSamples");
    const featureCount = resolveSubsetSize(this.maxFeatures, nFeatures, "maxFeatures");
    const trainingHasBothClasses = y.some((value) => value === 0) && y.some((value) => value === 1);

    const random =
      this.randomState === undefined
        ? Math.random
        : (() => {
            const rng = new Mulberry32(this.randomState!);
            return () => rng.next();
          })();

    this.estimators_ = new Array<ClassifierLike>(this.nEstimators);
    this.estimatorsFeatures_ = new Array<number[]>(this.nEstimators);

    for (let estimatorIndex = 0; estimatorIndex < this.nEstimators; estimatorIndex += 1) {
      let sampleIndices = this.bootstrap
        ? sampleWithReplacement(sampleCount, nSamples, random)
        : sampleWithoutReplacement(sampleCount, nSamples, random);
      if (trainingHasBothClasses && sampleCount >= 2 && !hasBothClasses(y, sampleIndices)) {
        for (let retry = 0; retry < 16; retry += 1) {
          sampleIndices = this.bootstrap
            ? sampleWithReplacement(sampleCount, nSamples, random)
            : sampleWithoutReplacement(sampleCount, nSamples, random);
          if (hasBothClasses(y, sampleIndices)) {
            break;
          }
        }
      }

      const featureIndices = this.bootstrapFeatures
        ? sampleWithReplacement(featureCount, nFeatures, random)
        : sampleWithoutReplacement(featureCount, nFeatures, random);

      const estimator = resolveEstimator(this.estimatorFactory);
      const XSubset = subsetRowsAndColumns(X, sampleIndices, featureIndices);
      const ySubset = new Array<number>(sampleIndices.length);
      for (let i = 0; i < sampleIndices.length; i += 1) {
        ySubset[i] = y[sampleIndices[i]];
      }
      estimator.fit(XSubset, ySubset);
      this.estimators_[estimatorIndex] = estimator;
      this.estimatorsFeatures_[estimatorIndex] = featureIndices;
    }

    this.nFeaturesIn_ = nFeatures;
    return this;
  }

  predict(X: Matrix): Vector {
    this.assertFitted();
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }

    const voteCounts = new Array<number>(X.length).fill(0);
    for (let estimatorIndex = 0; estimatorIndex < this.estimators_.length; estimatorIndex += 1) {
      const estimator = this.estimators_[estimatorIndex];
      const featureIndices = this.estimatorsFeatures_[estimatorIndex];
      const XProjected = subsetColumns(X, featureIndices);
      const predictions = estimator.predict(XProjected);
      for (let i = 0; i < predictions.length; i += 1) {
        if (predictions[i] === 1) {
          voteCounts[i] += 1;
        }
      }
    }

    const threshold = this.estimators_.length;
    return voteCounts.map((votes) => (votes * 2 >= threshold ? 1 : 0));
  }

  predictProba(X: Matrix): Matrix {
    this.assertFitted();
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }

    const positive = new Array<number>(X.length).fill(0);
    for (let estimatorIndex = 0; estimatorIndex < this.estimators_.length; estimatorIndex += 1) {
      const estimator = this.estimators_[estimatorIndex];
      const featureIndices = this.estimatorsFeatures_[estimatorIndex];
      const XProjected = subsetColumns(X, featureIndices);
      if (typeof estimator.predictProba === "function") {
        const proba = estimator.predictProba(XProjected);
        for (let i = 0; i < proba.length; i += 1) {
          positive[i] += proba[i][1];
        }
      } else {
        const pred = estimator.predict(XProjected);
        for (let i = 0; i < pred.length; i += 1) {
          positive[i] += pred[i];
        }
      }
    }

    return positive.map((sum) => {
      const p = sum / this.estimators_.length;
      return [1 - p, p];
    });
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return accuracyScore(y, this.predict(X));
  }

  private assertFitted(): void {
    if (this.estimators_.length === 0 || this.nFeaturesIn_ === null) {
      throw new Error("BaggingClassifier has not been fitted.");
    }
  }
}
