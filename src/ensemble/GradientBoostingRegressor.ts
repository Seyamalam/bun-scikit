import type { Matrix, Vector } from "../types";
import { r2Score } from "../metrics/regression";
import { DecisionTreeRegressor } from "../tree/DecisionTreeRegressor";
import { assertFiniteVector, validateRegressionInputs } from "../utils/validation";

export interface GradientBoostingRegressorOptions {
  nEstimators?: number;
  learningRate?: number;
  maxDepth?: number;
  minSamplesSplit?: number;
  minSamplesLeaf?: number;
  subsample?: number;
  randomState?: number;
}

function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function sampleWithoutReplacement(count: number, size: number, random: () => number): number[] {
  const pool = Array.from({ length: size }, (_, i) => i);
  for (let i = pool.length - 1; i > 0; i -= 1) {
    const j = Math.floor(random() * (i + 1));
    const tmp = pool[i];
    pool[i] = pool[j];
    pool[j] = tmp;
  }
  return pool.slice(0, count);
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

export class GradientBoostingRegressor {
  estimators_: DecisionTreeRegressor[] = [];
  init_: number | null = null;
  featureImportances_: Vector | null = null;

  private readonly nEstimators: number;
  private readonly learningRate: number;
  private readonly maxDepth: number;
  private readonly minSamplesSplit: number;
  private readonly minSamplesLeaf: number;
  private readonly subsample: number;
  private readonly randomState?: number;
  private isFitted = false;

  constructor(options: GradientBoostingRegressorOptions = {}) {
    this.nEstimators = options.nEstimators ?? 100;
    this.learningRate = options.learningRate ?? 0.1;
    this.maxDepth = options.maxDepth ?? 3;
    this.minSamplesSplit = options.minSamplesSplit ?? 2;
    this.minSamplesLeaf = options.minSamplesLeaf ?? 1;
    this.subsample = options.subsample ?? 1;
    this.randomState = options.randomState;

    if (!Number.isInteger(this.nEstimators) || this.nEstimators < 1) {
      throw new Error(`nEstimators must be an integer >= 1. Got ${this.nEstimators}.`);
    }
    if (!Number.isFinite(this.learningRate) || this.learningRate <= 0) {
      throw new Error(`learningRate must be finite and > 0. Got ${this.learningRate}.`);
    }
    if (!Number.isFinite(this.subsample) || this.subsample <= 0 || this.subsample > 1) {
      throw new Error(`subsample must be in (0, 1]. Got ${this.subsample}.`);
    }
  }

  fit(X: Matrix, y: Vector, sampleWeight?: Vector): this {
    validateRegressionInputs(X, y);
    const nSamples = X.length;
    const random = this.randomState === undefined ? Math.random : mulberry32(this.randomState);

    let mean = 0;
    for (let i = 0; i < y.length; i += 1) {
      mean += y[i];
    }
    mean /= y.length;
    this.init_ = mean;
    this.estimators_ = [];
    this.featureImportances_ = null;

    const currentPrediction = new Array<number>(nSamples).fill(mean);
    const sampleCount = Math.max(1, Math.floor(this.subsample * nSamples));

    for (let t = 0; t < this.nEstimators; t += 1) {
      const residuals = new Array<number>(nSamples);
      for (let i = 0; i < nSamples; i += 1) {
        residuals[i] = y[i] - currentPrediction[i];
      }

      const indices =
        sampleCount === nSamples
          ? Array.from({ length: nSamples }, (_, i) => i)
          : sampleWithoutReplacement(sampleCount, nSamples, random);

      const tree = new DecisionTreeRegressor({
        maxDepth: this.maxDepth,
        minSamplesSplit: this.minSamplesSplit,
        minSamplesLeaf: this.minSamplesLeaf,
        randomState: this.randomState === undefined ? undefined : this.randomState + t + 1,
      });
      tree.fit(subsetMatrix(X, indices), subsetVector(residuals, indices));
      const update = tree.predict(X);
      for (let i = 0; i < nSamples; i += 1) {
        currentPrediction[i] += this.learningRate * update[i];
      }
      this.estimators_.push(tree);
    }

    this.computeFeatureImportances(X[0].length);
    this.isFitted = true;
    return this;
  }

  predict(X: Matrix): Vector {
    this.assertFitted();
    const out = new Array<number>(X.length).fill(this.init_!);
    for (let t = 0; t < this.estimators_.length; t += 1) {
      const update = this.estimators_[t].predict(X);
      for (let i = 0; i < out.length; i += 1) {
        out[i] += this.learningRate * update[i];
      }
    }
    return out;
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return r2Score(y, this.predict(X));
  }

  private assertFitted(): void {
    if (!this.isFitted || this.init_ === null) {
      throw new Error("GradientBoostingRegressor has not been fitted.");
    }
  }

  private computeFeatureImportances(featureCount: number): void {
    const raw = new Array<number>(featureCount).fill(0);
    for (let i = 0; i < this.estimators_.length; i += 1) {
      const importances = this.estimators_[i].featureImportances_;
      if (!importances) {
        continue;
      }
      for (let j = 0; j < featureCount; j += 1) {
        raw[j] += importances[j];
      }
    }
    let sum = 0;
    for (let i = 0; i < raw.length; i += 1) {
      sum += raw[i];
    }
    this.featureImportances_ =
      sum > 0 ? raw.map((value) => value / sum) : new Array<number>(featureCount).fill(0);
  }
}

