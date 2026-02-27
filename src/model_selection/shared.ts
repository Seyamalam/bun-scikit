import type { Matrix, Vector } from "../types";
import { accuracyScore, f1Score, precisionScore, recallScore } from "../metrics/classification";
import { meanSquaredError, r2Score } from "../metrics/regression";
import { assertFiniteMatrix, assertFiniteVector, assertVectorLength } from "../utils/validation";
import { KFold, type FoldIndices } from "./KFold";
import { StratifiedKFold } from "./StratifiedKFold";

export type BuiltInScoring =
  | "accuracy"
  | "f1"
  | "precision"
  | "recall"
  | "r2"
  | "mean_squared_error"
  | "neg_mean_squared_error";

export type ScoringFn = (yTrue: Vector, yPred: Vector) => number;

export interface CrossValEstimator {
  fit(X: Matrix, y: Vector, ...extraFitArgs: unknown[]): unknown;
  predict(X: Matrix): Vector;
  score?(X: Matrix, y: Vector): number;
}

export type CrossValSplitter = {
  split(X: Matrix, y?: Vector, groups?: Vector): FoldIndices[];
};

export function isBinaryVector(y: Vector): boolean {
  for (let i = 0; i < y.length; i += 1) {
    const value = y[i];
    if (!(value === 0 || value === 1)) {
      return false;
    }
  }
  return true;
}

export function subsetMatrix(X: Matrix, indices: number[]): Matrix {
  const out = new Array(indices.length);
  for (let i = 0; i < indices.length; i += 1) {
    out[i] = X[indices[i]];
  }
  return out;
}

export function subsetVector(y: Vector, indices: number[]): Vector {
  const out = new Array(indices.length);
  for (let i = 0; i < indices.length; i += 1) {
    out[i] = y[indices[i]];
  }
  return out;
}

export function resolveBuiltInScorer(scoring: BuiltInScoring): ScoringFn {
  switch (scoring) {
    case "accuracy":
      return accuracyScore;
    case "f1":
      return f1Score;
    case "precision":
      return precisionScore;
    case "recall":
      return recallScore;
    case "r2":
      return r2Score;
    case "mean_squared_error":
      return (yTrue, yPred) => meanSquaredError(yTrue, yPred) as number;
    case "neg_mean_squared_error":
      return (yTrue, yPred) => -(meanSquaredError(yTrue, yPred) as number);
    default: {
      const exhaustive: never = scoring;
      throw new Error(`Unsupported scoring metric: ${exhaustive}`);
    }
  }
}

export function resolveFolds(
  X: Matrix,
  y: Vector,
  cv: number | CrossValSplitter | undefined,
  groups?: Vector,
): FoldIndices[] {
  if (typeof cv === "number") {
    if (!Number.isInteger(cv) || cv < 2) {
      throw new Error(`cv must be an integer >= 2. Got ${cv}.`);
    }
    if (isBinaryVector(y)) {
      return new StratifiedKFold({ nSplits: cv, shuffle: false }).split(X, y);
    }
    return new KFold({ nSplits: cv, shuffle: false }).split(X, y);
  }

  if (cv) {
    return cv.split(X, y, groups);
  }

  if (isBinaryVector(y)) {
    return new StratifiedKFold({ nSplits: 5, shuffle: false }).split(X, y);
  }
  return new KFold({ nSplits: 5, shuffle: false }).split(X, y);
}

export function validateCrossValInputs(
  X: Matrix,
  y: Vector,
  groups?: Vector,
  sampleWeight?: Vector,
): void {
  if (!Array.isArray(X) || X.length === 0) {
    throw new Error("X must be a non-empty matrix.");
  }
  assertFiniteMatrix(X);
  assertVectorLength(y, X.length);
  assertFiniteVector(y);
  if (groups) {
    assertVectorLength(groups, X.length);
    assertFiniteVector(groups);
  }
  if (sampleWeight) {
    assertVectorLength(sampleWeight, X.length);
    assertFiniteVector(sampleWeight);
    for (let i = 0; i < sampleWeight.length; i += 1) {
      if (sampleWeight[i] < 0) {
        throw new Error(`sampleWeight must contain non-negative values. Got ${sampleWeight[i]} at ${i}.`);
      }
    }
  }
}

export function evaluateEstimatorScore(
  estimator: CrossValEstimator,
  X: Matrix,
  y: Vector,
  scoring?: BuiltInScoring | ScoringFn,
): number {
  const scorer =
    typeof scoring === "function"
      ? scoring
      : scoring
        ? resolveBuiltInScorer(scoring)
        : null;
  if (scorer) {
    return scorer(y, estimator.predict(X));
  }
  if (typeof estimator.score === "function") {
    return estimator.score(X, y);
  }
  throw new Error(
    "Estimator must implement score() when no explicit scoring function is provided.",
  );
}

export function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function shuffleInPlace(values: number[], seed: number): void {
  const random = mulberry32(seed);
  for (let i = values.length - 1; i > 0; i -= 1) {
    const j = Math.floor(random() * (i + 1));
    const tmp = values[i];
    values[i] = values[j];
    values[j] = tmp;
  }
}
