import type { Matrix, Vector } from "../types";
import { accuracyScore } from "../metrics/classification";
import { assertFiniteVector, validateBinaryClassificationInputs } from "../utils/validation";
import { DecisionTreeClassifier } from "../tree/DecisionTreeClassifier";

type ClassifierLike = {
  fit(X: Matrix, y: Vector): unknown;
  predict(X: Matrix): Vector;
};

export interface AdaBoostClassifierOptions {
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

export class AdaBoostClassifier {
  classes_: Vector = [0, 1];
  estimators_: ClassifierLike[] = [];
  estimatorWeights_: Vector = [];

  private readonly estimatorFactory: () => ClassifierLike;
  private readonly nEstimators: number;
  private readonly learningRate: number;
  private readonly randomState?: number;
  private isFitted = false;

  constructor(
    estimatorFactory: (() => ClassifierLike) | null = null,
    options: AdaBoostClassifierOptions = {},
  ) {
    this.estimatorFactory =
      estimatorFactory ??
      (() => new DecisionTreeClassifier({ maxDepth: 1, minSamplesLeaf: 1, minSamplesSplit: 2 }));
    this.nEstimators = options.nEstimators ?? 50;
    this.learningRate = options.learningRate ?? 1.0;
    this.randomState = options.randomState;

    if (!Number.isInteger(this.nEstimators) || this.nEstimators < 1) {
      throw new Error(`nEstimators must be an integer >= 1. Got ${this.nEstimators}.`);
    }
    if (!Number.isFinite(this.learningRate) || this.learningRate <= 0) {
      throw new Error(`learningRate must be finite and > 0. Got ${this.learningRate}.`);
    }
  }

  fit(X: Matrix, y: Vector): this {
    validateBinaryClassificationInputs(X, y);
    const nSamples = X.length;
    const random = this.randomState === undefined ? Math.random : mulberry32(this.randomState);
    const weights = new Array<number>(nSamples).fill(1 / nSamples);

    this.estimators_ = [];
    this.estimatorWeights_ = [];

    for (let t = 0; t < this.nEstimators; t += 1) {
      const sampleIndices = weightedSampleIndices(weights, nSamples, random);

      let hasZero = false;
      let hasOne = false;
      for (let i = 0; i < sampleIndices.length; i += 1) {
        if (y[sampleIndices[i]] === 0) {
          hasZero = true;
        } else {
          hasOne = true;
        }
        if (hasZero && hasOne) {
          break;
        }
      }
      if (!hasZero || !hasOne) {
        const missingClass = hasZero ? 1 : 0;
        let replacementIndex = -1;
        let bestWeight = -Infinity;
        for (let i = 0; i < nSamples; i += 1) {
          if (y[i] === missingClass && weights[i] > bestWeight) {
            bestWeight = weights[i];
            replacementIndex = i;
          }
        }
        if (replacementIndex !== -1) {
          sampleIndices[sampleIndices.length - 1] = replacementIndex;
        }
      }

      const estimator = this.estimatorFactory();
      estimator.fit(subsetMatrix(X, sampleIndices), subsetVector(y, sampleIndices));
      const pred = estimator.predict(X);

      let weightedError = 0;
      for (let i = 0; i < nSamples; i += 1) {
        if (pred[i] !== y[i]) {
          weightedError += weights[i];
        }
      }

      const err = Math.min(1 - 1e-12, Math.max(1e-12, weightedError));
      if (err >= 0.5) {
        break;
      }
      const alpha = this.learningRate * 0.5 * Math.log((1 - err) / err);

      let weightSum = 0;
      for (let i = 0; i < nSamples; i += 1) {
        const yi = y[i] === 1 ? 1 : -1;
        const pi = pred[i] === 1 ? 1 : -1;
        weights[i] *= Math.exp(-alpha * yi * pi);
        weightSum += weights[i];
      }
      for (let i = 0; i < nSamples; i += 1) {
        weights[i] /= weightSum;
      }

      this.estimators_.push(estimator);
      this.estimatorWeights_.push(alpha);
    }

    if (this.estimators_.length === 0) {
      throw new Error("AdaBoostClassifier failed to train any weak learner.");
    }

    this.isFitted = true;
    return this;
  }

  decisionFunction(X: Matrix): Vector {
    this.assertFitted();
    const scores = new Array<number>(X.length).fill(0);
    for (let t = 0; t < this.estimators_.length; t += 1) {
      const pred = this.estimators_[t].predict(X);
      const alpha = this.estimatorWeights_[t];
      for (let i = 0; i < pred.length; i += 1) {
        scores[i] += alpha * (pred[i] === 1 ? 1 : -1);
      }
    }
    return scores;
  }

  predictProba(X: Matrix): Matrix {
    const scores = this.decisionFunction(X);
    return scores.map((score) => {
      const p1 = 1 / (1 + Math.exp(-2 * score));
      return [1 - p1, p1];
    });
  }

  predict(X: Matrix): Vector {
    return this.decisionFunction(X).map((score) => (score >= 0 ? 1 : 0));
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return accuracyScore(y, this.predict(X));
  }

  private assertFitted(): void {
    if (!this.isFitted || this.estimators_.length === 0) {
      throw new Error("AdaBoostClassifier has not been fitted.");
    }
  }
}
