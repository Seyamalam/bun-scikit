import type { Matrix, Vector } from "../types";
import { r2Score } from "../metrics/regression";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  assertNonEmptyMatrix,
  validateRegressionInputs,
} from "../utils/validation";

type RegressorLike = {
  fit(X: Matrix, y: Vector, sampleWeight?: Vector): unknown;
  predict(X: Matrix): Vector;
};

export interface BaggingRegressorOptions {
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

function resolveEstimator(input: (() => RegressorLike) | RegressorLike): RegressorLike {
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

export class BaggingRegressor {
  estimators_: RegressorLike[] = [];
  estimatorsFeatures_: number[][] = [];

  private estimatorFactory: (() => RegressorLike) | RegressorLike;
  private nEstimators: number;
  private maxSamples?: number;
  private maxFeatures?: number;
  private bootstrap: boolean;
  private bootstrapFeatures: boolean;
  private randomState?: number;
  private nFeaturesIn_: number | null = null;

  constructor(
    estimatorFactory: (() => RegressorLike) | RegressorLike,
    options: BaggingRegressorOptions = {},
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

    const nSamples = X.length;
    const nFeatures = X[0].length;
    const sampleCount = resolveSubsetSize(this.maxSamples, nSamples, "maxSamples");
    const featureCount = resolveSubsetSize(this.maxFeatures, nFeatures, "maxFeatures");

    const random =
      this.randomState === undefined
        ? Math.random
        : (() => {
            const rng = new Mulberry32(this.randomState!);
            return () => rng.next();
          })();

    this.estimators_ = new Array<RegressorLike>(this.nEstimators);
    this.estimatorsFeatures_ = new Array<number[]>(this.nEstimators);

    for (let estimatorIndex = 0; estimatorIndex < this.nEstimators; estimatorIndex += 1) {
      const sampleIndices = this.bootstrap
        ? sampleWithReplacement(sampleCount, nSamples, random)
        : sampleWithoutReplacement(sampleCount, nSamples, random);

      const featureIndices = this.bootstrapFeatures
        ? sampleWithReplacement(featureCount, nFeatures, random)
        : sampleWithoutReplacement(featureCount, nFeatures, random);

      const estimator = resolveEstimator(this.estimatorFactory);
      const XSubset = subsetRowsAndColumns(X, sampleIndices, featureIndices);
      const ySubset = new Array<number>(sampleIndices.length);
      const sampleWeightSubset = sampleWeight ? new Array<number>(sampleIndices.length) : undefined;
      for (let i = 0; i < sampleIndices.length; i += 1) {
        ySubset[i] = y[sampleIndices[i]];
        if (sampleWeightSubset) {
          sampleWeightSubset[i] = sampleWeight![sampleIndices[i]];
        }
      }
      estimator.fit(XSubset, ySubset, sampleWeightSubset);
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

    const totals = new Array<number>(X.length).fill(0);
    for (let estimatorIndex = 0; estimatorIndex < this.estimators_.length; estimatorIndex += 1) {
      const estimator = this.estimators_[estimatorIndex];
      const featureIndices = this.estimatorsFeatures_[estimatorIndex];
      const XProjected = subsetColumns(X, featureIndices);
      const predictions = estimator.predict(XProjected);
      for (let i = 0; i < predictions.length; i += 1) {
        totals[i] += predictions[i];
      }
    }
    return totals.map((value) => value / this.estimators_.length);
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return r2Score(y, this.predict(X)) as number;
  }

  private assertFitted(): void {
    if (this.estimators_.length === 0 || this.nFeaturesIn_ === null) {
      throw new Error("BaggingRegressor has not been fitted.");
    }
  }
}
