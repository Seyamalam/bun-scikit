import type { Matrix, Vector } from "../types";
import { accuracyScore } from "../metrics/classification";
import { DecisionTreeRegressor } from "../tree/DecisionTreeRegressor";
import { assertFiniteVector, validateBinaryClassificationInputs } from "../utils/validation";

export interface GradientBoostingClassifierOptions {
  nEstimators?: number;
  learningRate?: number;
  maxDepth?: number;
  minSamplesSplit?: number;
  minSamplesLeaf?: number;
  subsample?: number;
  randomState?: number;
}

function sigmoid(z: number): number {
  if (z >= 0) {
    const expNeg = Math.exp(-z);
    return 1 / (1 + expNeg);
  }
  const expPos = Math.exp(z);
  return expPos / (1 + expPos);
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

export class GradientBoostingClassifier {
  classes_: Vector = [0, 1];
  estimators_: DecisionTreeRegressor[] = [];
  init_: number | null = null;

  private readonly nEstimators: number;
  private readonly learningRate: number;
  private readonly maxDepth: number;
  private readonly minSamplesSplit: number;
  private readonly minSamplesLeaf: number;
  private readonly subsample: number;
  private readonly randomState?: number;
  private isFitted = false;

  constructor(options: GradientBoostingClassifierOptions = {}) {
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

  fit(X: Matrix, y: Vector): this {
    validateBinaryClassificationInputs(X, y);
    const nSamples = X.length;
    const random = this.randomState === undefined ? Math.random : mulberry32(this.randomState);

    let positive = 0;
    for (let i = 0; i < nSamples; i += 1) {
      positive += y[i];
    }
    const p = Math.min(1 - 1e-12, Math.max(1e-12, positive / nSamples));
    this.init_ = Math.log(p / (1 - p));
    this.estimators_ = [];

    const logits = new Array<number>(nSamples).fill(this.init_);
    const sampleCount = Math.max(1, Math.floor(this.subsample * nSamples));

    for (let t = 0; t < this.nEstimators; t += 1) {
      const gradients = new Array<number>(nSamples);
      for (let i = 0; i < nSamples; i += 1) {
        gradients[i] = y[i] - sigmoid(logits[i]);
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
      tree.fit(subsetMatrix(X, indices), subsetVector(gradients, indices));

      const update = tree.predict(X);
      for (let i = 0; i < nSamples; i += 1) {
        logits[i] += this.learningRate * update[i];
      }
      this.estimators_.push(tree);
    }

    this.isFitted = true;
    return this;
  }

  decisionFunction(X: Matrix): Vector {
    this.assertFitted();
    const logits = new Array<number>(X.length).fill(this.init_!);
    for (let t = 0; t < this.estimators_.length; t += 1) {
      const update = this.estimators_[t].predict(X);
      for (let i = 0; i < logits.length; i += 1) {
        logits[i] += this.learningRate * update[i];
      }
    }
    return logits;
  }

  predictProba(X: Matrix): Matrix {
    return this.decisionFunction(X).map((logit) => {
      const p1 = sigmoid(logit);
      return [1 - p1, p1];
    });
  }

  predict(X: Matrix): Vector {
    return this.predictProba(X).map((row) => (row[1] >= 0.5 ? 1 : 0));
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return accuracyScore(y, this.predict(X));
  }

  private assertFitted(): void {
    if (!this.isFitted || this.init_ === null) {
      throw new Error("GradientBoostingClassifier has not been fitted.");
    }
  }
}
