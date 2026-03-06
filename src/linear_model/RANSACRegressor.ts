import type { Matrix, RegressionModel, Vector } from "../types";
import { r2Score } from "../metrics/regression";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  validateRegressionInputs,
} from "../utils/validation";
import { LinearRegression } from "./LinearRegression";

export interface RANSACRegressorOptions {
  minSamples?: number;
  residualThreshold?: number;
  maxTrials?: number;
  stopNInliers?: number;
  randomState?: number;
  estimatorFactory?: () => RegressionModel;
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

function sampleWithoutReplacement(count: number, choose: number, random: () => number): number[] {
  const pool = Array.from({ length: count }, (_, index) => index);
  for (let i = pool.length - 1; i > 0; i -= 1) {
    const j = Math.floor(random() * (i + 1));
    const tmp = pool[i]!;
    pool[i] = pool[j]!;
    pool[j] = tmp;
  }
  return pool.slice(0, choose);
}

function subsetMatrix(X: Matrix, indices: number[]): Matrix {
  return indices.map((index) => X[index]!.slice());
}

function subsetVector(y: Vector, indices: number[]): Vector {
  return indices.map((index) => y[index]!);
}

export class RANSACRegressor implements RegressionModel {
  estimator_: RegressionModel | null = null;
  inlierMask_: boolean[] | null = null;
  nTrials_ = 0;

  private readonly minSamples?: number;
  private readonly residualThreshold?: number;
  private readonly maxTrials: number;
  private readonly stopNInliers?: number;
  private readonly randomState: number;
  private readonly estimatorFactory: () => RegressionModel;

  constructor(options: RANSACRegressorOptions = {}) {
    this.minSamples = options.minSamples;
    this.residualThreshold = options.residualThreshold;
    this.maxTrials = options.maxTrials ?? 100;
    this.stopNInliers = options.stopNInliers;
    this.randomState = options.randomState ?? 42;
    this.estimatorFactory = options.estimatorFactory ?? (() => new LinearRegression());

    if (this.minSamples !== undefined && (!Number.isInteger(this.minSamples) || this.minSamples < 1)) {
      throw new Error(`minSamples must be an integer >= 1. Got ${this.minSamples}.`);
    }
    if (
      this.residualThreshold !== undefined &&
      (!Number.isFinite(this.residualThreshold) || this.residualThreshold < 0)
    ) {
      throw new Error(
        `residualThreshold must be finite and >= 0. Got ${this.residualThreshold}.`,
      );
    }
    if (!Number.isInteger(this.maxTrials) || this.maxTrials < 1) {
      throw new Error(`maxTrials must be an integer >= 1. Got ${this.maxTrials}.`);
    }
    if (this.stopNInliers !== undefined && (!Number.isInteger(this.stopNInliers) || this.stopNInliers < 1)) {
      throw new Error(`stopNInliers must be an integer >= 1. Got ${this.stopNInliers}.`);
    }
  }

  fit(X: Matrix, y: Vector, sampleWeight?: Vector): this {
    validateRegressionInputs(X, y);

    const minSamples = this.minSamples ?? Math.max(2, X[0]!.length + 1);
    if (minSamples > X.length) {
      throw new Error(`minSamples (${minSamples}) cannot exceed sample count (${X.length}).`);
    }

    const residualThreshold = this.residualThreshold ?? this.estimateResidualThreshold(y);
    const random = mulberry32(this.randomState);

    let bestInlierCount = -1;
    let bestInlierMask: boolean[] | null = null;
    let bestEstimator: RegressionModel | null = null;
    let bestResidualSum = Number.POSITIVE_INFINITY;

    for (let trial = 0; trial < this.maxTrials; trial += 1) {
      this.nTrials_ = trial + 1;
      const subsetIndices = sampleWithoutReplacement(X.length, minSamples, random);
      const candidate = this.estimatorFactory();

      try {
        candidate.fit(subsetMatrix(X, subsetIndices), subsetVector(y, subsetIndices));
      } catch {
        continue;
      }

      let residualSum = 0;
      let inlierCount = 0;
      const predictions = candidate.predict(X);
      const inlierMask = new Array<boolean>(X.length).fill(false);

      for (let i = 0; i < X.length; i += 1) {
        const residual = Math.abs(y[i]! - predictions[i]!);
        residualSum += residual;
        if (residual <= residualThreshold) {
          inlierMask[i] = true;
          inlierCount += 1;
        }
      }

      if (
        inlierCount > bestInlierCount ||
        (inlierCount === bestInlierCount && residualSum < bestResidualSum)
      ) {
        bestInlierCount = inlierCount;
        bestInlierMask = inlierMask;
        bestResidualSum = residualSum;
        bestEstimator = candidate;
      }

      if (this.stopNInliers !== undefined && inlierCount >= this.stopNInliers) {
        break;
      }
    }

    if (!bestInlierMask || bestInlierCount < minSamples) {
      throw new Error("RANSACRegressor could not find a valid consensus set.");
    }

    const inlierIndices: number[] = [];
    for (let i = 0; i < bestInlierMask.length; i += 1) {
      if (bestInlierMask[i]!) {
        inlierIndices.push(i);
      }
    }

    const finalEstimator = this.estimatorFactory();
    finalEstimator.fit(subsetMatrix(X, inlierIndices), subsetVector(y, inlierIndices));

    this.estimator_ = finalEstimator;
    this.inlierMask_ = bestInlierMask;
    return this;
  }

  predict(X: Matrix): Vector {
    this.assertFitted();
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    return this.estimator_!.predict(X);
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return r2Score(y, this.predict(X));
  }

  private estimateResidualThreshold(y: Vector): number {
    const sorted = y.slice().sort((left, right) => left - right);
    const median = sorted[Math.floor(sorted.length / 2)]!;
    const deviations = sorted.map((value) => Math.abs(value - median)).sort((left, right) => left - right);
    const mad = deviations[Math.floor(deviations.length / 2)]!;
    return mad === 0 ? 1 : 2.5 * mad;
  }

  private assertFitted(): void {
    if (!this.estimator_ || !this.inlierMask_) {
      throw new Error("RANSACRegressor has not been fitted.");
    }
  }
}
