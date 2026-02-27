import type { Matrix, Vector } from "../types";
import {
  accuracyScore,
  f1Score,
  precisionScore,
  recallScore,
} from "../metrics/classification";
import {
  explainedVarianceScore,
  meanAbsoluteError,
  meanAbsolutePercentageError,
  meanSquaredError,
  r2Score,
} from "../metrics/regression";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  assertVectorLength,
} from "../utils/validation";
import type { BuiltInScoring, ScoringFn } from "../model_selection/shared";

export interface PermutationImportanceEstimator {
  predict(X: Matrix): Vector;
  score?(X: Matrix, y: Vector): number;
}

export interface PermutationImportanceOptions {
  scoring?: BuiltInScoring | ScoringFn;
  nRepeats?: number;
  randomState?: number;
  sampleWeight?: Vector;
}

export interface PermutationImportanceResult {
  importances: number[][];
  importancesMean: number[];
  importancesStd: number[];
  baselineScore: number;
  scoring: string;
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

function shuffleInPlace(values: number[], random: () => number): void {
  for (let i = values.length - 1; i > 0; i -= 1) {
    const j = Math.floor(random() * (i + 1));
    const tmp = values[i];
    values[i] = values[j];
    values[j] = tmp;
  }
}

function cloneMatrix(X: Matrix): Matrix {
  return X.map((row) => row.slice());
}

function scorerName(scoring: BuiltInScoring | ScoringFn | undefined): string {
  if (!scoring) {
    return "estimator.score";
  }
  return typeof scoring === "function" ? "custom" : scoring;
}

function isLossMetric(scoring: BuiltInScoring | ScoringFn | undefined): boolean {
  return scoring === "mean_squared_error";
}

function resolveBuiltInScore(
  scoring: BuiltInScoring,
  yTrue: Vector,
  yPred: Vector,
  sampleWeight?: Vector,
): number {
  switch (scoring) {
    case "accuracy":
      return accuracyScore(yTrue, yPred, sampleWeight);
    case "f1":
      return f1Score(yTrue, yPred, 1, sampleWeight);
    case "precision":
      return precisionScore(yTrue, yPred, 1, sampleWeight);
    case "recall":
      return recallScore(yTrue, yPred, 1, sampleWeight);
    case "r2":
      return r2Score(yTrue, yPred, { sampleWeight }) as number;
    case "mean_squared_error":
      return meanSquaredError(yTrue, yPred, { sampleWeight }) as number;
    case "neg_mean_squared_error":
      return -(meanSquaredError(yTrue, yPred, { sampleWeight }) as number);
    default: {
      const exhaustive: never = scoring;
      throw new Error(`Unsupported scoring metric: ${exhaustive}`);
    }
  }
}

function evaluateScore(
  estimator: PermutationImportanceEstimator,
  X: Matrix,
  y: Vector,
  scoring: BuiltInScoring | ScoringFn | undefined,
  sampleWeight?: Vector,
): number {
  if (scoring) {
    const yPred = estimator.predict(X);
    if (typeof scoring === "function") {
      return scoring(y, yPred);
    }
    return resolveBuiltInScore(scoring, y, yPred, sampleWeight);
  }
  if (typeof estimator.score === "function") {
    return estimator.score(X, y);
  }
  throw new Error(
    "Estimator must implement score() when scoring is not provided.",
  );
}

function mean(values: number[]): number {
  let sum = 0;
  for (let i = 0; i < values.length; i += 1) {
    sum += values[i];
  }
  return sum / Math.max(1, values.length);
}

function std(values: number[]): number {
  if (values.length < 2) {
    return 0;
  }
  const avg = mean(values);
  let sumSq = 0;
  for (let i = 0; i < values.length; i += 1) {
    const d = values[i] - avg;
    sumSq += d * d;
  }
  return Math.sqrt(sumSq / values.length);
}

export function permutationImportance(
  estimator: PermutationImportanceEstimator,
  X: Matrix,
  y: Vector,
  options: PermutationImportanceOptions = {},
): PermutationImportanceResult {
  if (typeof estimator !== "object" || estimator === null) {
    throw new Error("estimator must be an object with predict() and optional score().");
  }
  if (typeof estimator.predict !== "function") {
    throw new Error("estimator must implement predict().");
  }
  if (!Array.isArray(X) || X.length === 0) {
    throw new Error("X must be a non-empty matrix.");
  }

  assertConsistentRowSize(X);
  assertFiniteMatrix(X);
  assertVectorLength(y, X.length);
  assertFiniteVector(y);
  if (options.sampleWeight) {
    assertVectorLength(options.sampleWeight, X.length);
    assertFiniteVector(options.sampleWeight);
  }

  const nRepeats = options.nRepeats ?? 5;
  if (!Number.isInteger(nRepeats) || nRepeats < 1) {
    throw new Error(`nRepeats must be an integer >= 1. Got ${nRepeats}.`);
  }
  const randomState = options.randomState ?? 42;
  if (!Number.isInteger(randomState)) {
    throw new Error(`randomState must be an integer. Got ${randomState}.`);
  }

  const nFeatures = X[0].length;
  const baselineScore = evaluateScore(
    estimator,
    X,
    y,
    options.scoring,
    options.sampleWeight,
  );
  const importanceSign = isLossMetric(options.scoring) ? -1 : 1;
  const importances = new Array<number[]>(nFeatures);

  for (let featureIndex = 0; featureIndex < nFeatures; featureIndex += 1) {
    const current = new Array<number>(nRepeats);
    for (let repeat = 0; repeat < nRepeats; repeat += 1) {
      const XPermuted = cloneMatrix(X);
      const rowOrder = Array.from({ length: X.length }, (_, idx) => idx);
      const random = mulberry32(
        randomState + featureIndex * 104_729 + repeat * 1_299_721,
      );
      shuffleInPlace(rowOrder, random);
      for (let row = 0; row < XPermuted.length; row += 1) {
        XPermuted[row][featureIndex] = X[rowOrder[row]][featureIndex];
      }
      const permutedScore = evaluateScore(
        estimator,
        XPermuted,
        y,
        options.scoring,
        options.sampleWeight,
      );
      current[repeat] = (baselineScore - permutedScore) * importanceSign;
    }
    importances[featureIndex] = current;
  }

  const importancesMean = importances.map((values) => mean(values));
  const importancesStd = importances.map((values) => std(values));

  return {
    importances,
    importancesMean,
    importancesStd,
    baselineScore,
    scoring: scorerName(options.scoring),
  };
}

export type {
  BuiltInScoring,
  ScoringFn,
};
