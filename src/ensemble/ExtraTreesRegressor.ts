import type { Matrix, RegressionModel, Vector } from "../types";
import { r2Score } from "../metrics/regression";
import { DecisionTreeRegressor } from "../tree/DecisionTreeRegressor";
import { assertFiniteVector, validateRegressionInputs } from "../utils/validation";
import type { RandomForestRegressorOptions } from "./RandomForestRegressor";

export interface ExtraTreesRegressorOptions extends RandomForestRegressorOptions {}

function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function normalizeSampleWeight(sampleWeight: Vector): Vector {
  let total = 0;
  for (let i = 0; i < sampleWeight.length; i += 1) {
    const value = sampleWeight[i];
    if (!Number.isFinite(value) || value < 0) {
      throw new Error(`sampleWeight must contain finite values >= 0. Got ${value} at index ${i}.`);
    }
    total += value;
  }
  if (total <= 0) {
    throw new Error("sampleWeight must sum to a positive value.");
  }
  return sampleWeight.map((value) => value / total);
}

function weightedSampleIndices(size: number, weights: Vector, random: () => number): Uint32Array {
  const cdf = new Array<number>(weights.length);
  let running = 0;
  for (let i = 0; i < weights.length; i += 1) {
    running += weights[i];
    cdf[i] = running;
  }
  cdf[cdf.length - 1] = 1;

  const out = new Uint32Array(size);
  for (let i = 0; i < size; i += 1) {
    const draw = random();
    let left = 0;
    let right = cdf.length - 1;
    while (left < right) {
      const middle = (left + right) >> 1;
      if (cdf[middle] >= draw) {
        right = middle;
      } else {
        left = middle + 1;
      }
    }
    out[i] = left;
  }
  return out;
}

function uniformBootstrapIndices(size: number, random: () => number): Uint32Array {
  const out = new Uint32Array(size);
  for (let i = 0; i < size; i += 1) {
    out[i] = Math.floor(random() * size);
  }
  return out;
}

export class ExtraTreesRegressor implements RegressionModel {
  featureImportances_: Vector | null = null;

  private options: ExtraTreesRegressorOptions;
  private trees: DecisionTreeRegressor[] = [];

  constructor(options: ExtraTreesRegressorOptions = {}) {
    this.options = {
      nEstimators: options.nEstimators ?? 100,
      maxDepth: options.maxDepth ?? undefined,
      minSamplesSplit: options.minSamplesSplit ?? 2,
      minSamplesLeaf: options.minSamplesLeaf ?? 1,
      maxFeatures: options.maxFeatures ?? 1.0,
      bootstrap: options.bootstrap ?? false,
      randomState: options.randomState,
    };
  }

  getParams(): ExtraTreesRegressorOptions {
    return { ...this.options };
  }

  setParams(params: Partial<ExtraTreesRegressorOptions>): this {
    this.options = { ...this.options, ...params };
    this.trees = [];
    this.featureImportances_ = null;
    return this;
  }

  fit(X: Matrix, y: Vector, sampleWeight?: Vector): this {
    validateRegressionInputs(X, y);
    if (sampleWeight && sampleWeight.length !== y.length) {
      throw new Error(`sampleWeight length mismatch. Expected ${y.length}, got ${sampleWeight.length}.`);
    }

    const random = this.options.randomState === undefined ? Math.random : mulberry32(this.options.randomState);
    const weights = sampleWeight ? normalizeSampleWeight(sampleWeight) : null;
    const nEstimators = this.options.nEstimators ?? 100;
    this.trees = new Array(nEstimators);

    for (let estimator = 0; estimator < nEstimators; estimator += 1) {
      let sampleIndices: Uint32Array | undefined;
      if (weights) {
        sampleIndices = weightedSampleIndices(X.length, weights, random);
      } else if (this.options.bootstrap) {
        sampleIndices = uniformBootstrapIndices(X.length, random);
      } else {
        sampleIndices = undefined;
      }

      const tree = new DecisionTreeRegressor({
        maxDepth: this.options.maxDepth,
        minSamplesSplit: this.options.minSamplesSplit,
        minSamplesLeaf: this.options.minSamplesLeaf,
        maxFeatures: this.options.maxFeatures,
        splitter: "random",
        randomState:
          this.options.randomState === undefined ? undefined : this.options.randomState + estimator + 1,
      });
      tree.fit(X, y, sampleIndices);
      this.trees[estimator] = tree;
    }

    this.computeFeatureImportances(X[0].length);
    return this;
  }

  predict(X: Matrix): Vector {
    if (this.trees.length === 0) {
      throw new Error("ExtraTreesRegressor has not been fitted.");
    }

    const sums = new Array<number>(X.length).fill(0);
    for (let i = 0; i < this.trees.length; i += 1) {
      const prediction = this.trees[i].predict(X);
      for (let j = 0; j < prediction.length; j += 1) {
        sums[j] += prediction[j];
      }
    }

    const out = new Array<number>(X.length);
    for (let i = 0; i < X.length; i += 1) {
      out[i] = sums[i] / this.trees.length;
    }
    return out;
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return r2Score(y, this.predict(X));
  }

  dispose(): void {
    this.trees = [];
  }

  private computeFeatureImportances(featureCount: number): void {
    if (this.trees.length === 0) {
      this.featureImportances_ = null;
      return;
    }

    const raw = new Array<number>(featureCount).fill(0);
    for (let i = 0; i < this.trees.length; i += 1) {
      const importances = this.trees[i].featureImportances_;
      if (!importances) {
        continue;
      }
      for (let j = 0; j < featureCount; j += 1) {
        raw[j] += importances[j];
      }
    }

    let total = 0;
    for (let i = 0; i < raw.length; i += 1) {
      total += raw[i];
    }
    this.featureImportances_ =
      total > 0 ? raw.map((value) => value / total) : new Array<number>(featureCount).fill(0);
  }
}
