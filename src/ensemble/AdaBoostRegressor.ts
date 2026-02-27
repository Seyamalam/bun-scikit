import type { Matrix, Vector } from "../types";
import { r2Score } from "../metrics/regression";
import { assertFiniteVector, validateRegressionInputs } from "../utils/validation";
import { DecisionTreeRegressor } from "../tree/DecisionTreeRegressor";

type RegressorLike = {
  fit(X: Matrix, y: Vector): unknown;
  predict(X: Matrix): Vector;
  featureImportances_?: Vector | null;
};

export interface AdaBoostRegressorOptions {
  nEstimators?: number;
  learningRate?: number;
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

function weightedSampleIndices(
  weights: Vector,
  count: number,
  random: () => number,
): number[] {
  const cdf = new Array<number>(weights.length);
  let cumulative = 0;
  for (let i = 0; i < weights.length; i += 1) {
    cumulative += weights[i];
    cdf[i] = cumulative;
  }
  const out = new Array<number>(count);
  for (let i = 0; i < count; i += 1) {
    const draw = random();
    let left = 0;
    let right = cdf.length - 1;
    while (left < right) {
      const mid = Math.floor((left + right) / 2);
      if (draw <= cdf[mid]) {
        right = mid;
      } else {
        left = mid + 1;
      }
    }
    out[i] = left;
  }
  return out;
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

export class AdaBoostRegressor {
  estimators_: RegressorLike[] = [];
  estimatorWeights_: Vector = [];
  featureImportances_: Vector | null = null;

  private estimatorFactory: () => RegressorLike;
  private nEstimators: number;
  private learningRate: number;
  private randomState?: number;
  private isFitted = false;

  constructor(
    estimatorFactory: (() => RegressorLike) | null = null,
    options: AdaBoostRegressorOptions = {},
  ) {
    this.estimatorFactory =
      estimatorFactory ?? (() => new DecisionTreeRegressor({ maxDepth: 3, minSamplesLeaf: 1 }));
    this.nEstimators = options.nEstimators ?? 50;
    this.learningRate = options.learningRate ?? 1.0;
    this.randomState = options.randomState;
    this.validateOptions();
  }

  getParams(): AdaBoostRegressorOptions {
    return {
      nEstimators: this.nEstimators,
      learningRate: this.learningRate,
      randomState: this.randomState,
    };
  }

  setParams(params: Partial<AdaBoostRegressorOptions>): this {
    if (params.nEstimators !== undefined) this.nEstimators = params.nEstimators;
    if (params.learningRate !== undefined) this.learningRate = params.learningRate;
    if (params.randomState !== undefined) this.randomState = params.randomState;
    this.validateOptions();
    return this;
  }

  fit(X: Matrix, y: Vector): this {
    validateRegressionInputs(X, y);
    const nSamples = X.length;
    const random = this.randomState === undefined ? Math.random : mulberry32(this.randomState);
    const weights = new Array<number>(nSamples).fill(1 / nSamples);

    this.estimators_ = [];
    this.estimatorWeights_ = [];
    this.featureImportances_ = null;

    for (let t = 0; t < this.nEstimators; t += 1) {
      const sampleIndices = weightedSampleIndices(weights, nSamples, random);
      const estimator = this.estimatorFactory();
      estimator.fit(subsetMatrix(X, sampleIndices), subsetVector(y, sampleIndices));
      const pred = estimator.predict(X);

      const absErrors = new Array<number>(nSamples);
      let maxError = 0;
      for (let i = 0; i < nSamples; i += 1) {
        const err = Math.abs(y[i] - pred[i]);
        absErrors[i] = err;
        if (err > maxError) {
          maxError = err;
        }
      }
      if (maxError <= 1e-12) {
        this.estimators_.push(estimator);
        this.estimatorWeights_.push(this.learningRate);
        break;
      }

      let weightedError = 0;
      const normalizedErrors = new Array<number>(nSamples);
      for (let i = 0; i < nSamples; i += 1) {
        const normalized = absErrors[i] / maxError;
        normalizedErrors[i] = normalized;
        weightedError += weights[i] * normalized;
      }

      const err = Math.max(1e-12, Math.min(1 - 1e-12, weightedError));
      if (err >= 0.5) {
        break;
      }
      const beta = err / (1 - err);
      const estimatorWeight = this.learningRate * Math.log(1 / beta);

      let weightSum = 0;
      for (let i = 0; i < nSamples; i += 1) {
        weights[i] *= beta ** (1 - normalizedErrors[i]);
        weightSum += weights[i];
      }
      for (let i = 0; i < nSamples; i += 1) {
        weights[i] /= weightSum;
      }

      this.estimators_.push(estimator);
      this.estimatorWeights_.push(estimatorWeight);
    }

    if (this.estimators_.length === 0) {
      throw new Error("AdaBoostRegressor failed to train any weak learner.");
    }

    this.computeFeatureImportances(X[0].length);
    this.isFitted = true;
    return this;
  }

  predict(X: Matrix): Vector {
    this.assertFitted();
    const sums = new Array<number>(X.length).fill(0);
    let totalWeight = 0;
    for (let t = 0; t < this.estimators_.length; t += 1) {
      const prediction = this.estimators_[t].predict(X);
      const weight = this.estimatorWeights_[t];
      totalWeight += weight;
      for (let i = 0; i < prediction.length; i += 1) {
        sums[i] += weight * prediction[i];
      }
    }
    if (totalWeight <= 0) {
      throw new Error("AdaBoostRegressor has invalid estimator weights.");
    }
    return sums.map((value) => value / totalWeight);
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return r2Score(y, this.predict(X));
  }

  private assertFitted(): void {
    if (!this.isFitted || this.estimators_.length === 0) {
      throw new Error("AdaBoostRegressor has not been fitted.");
    }
  }

  private computeFeatureImportances(featureCount: number): void {
    const raw = new Array<number>(featureCount).fill(0);
    let totalWeight = 0;
    for (let i = 0; i < this.estimators_.length; i += 1) {
      const importances = this.estimators_[i].featureImportances_;
      if (!importances) {
        continue;
      }
      const weight = Math.max(0, this.estimatorWeights_[i]);
      totalWeight += weight;
      for (let j = 0; j < featureCount; j += 1) {
        raw[j] += weight * importances[j];
      }
    }
    if (totalWeight <= 0) {
      this.featureImportances_ = new Array<number>(featureCount).fill(0);
      return;
    }
    let sum = 0;
    for (let i = 0; i < raw.length; i += 1) {
      sum += raw[i];
    }
    this.featureImportances_ =
      sum > 0 ? raw.map((value) => value / sum) : new Array<number>(featureCount).fill(0);
  }

  private validateOptions(): void {
    if (!Number.isInteger(this.nEstimators) || this.nEstimators < 1) {
      throw new Error(`nEstimators must be an integer >= 1. Got ${this.nEstimators}.`);
    }
    if (!Number.isFinite(this.learningRate) || this.learningRate <= 0) {
      throw new Error(`learningRate must be finite and > 0. Got ${this.learningRate}.`);
    }
  }
}
