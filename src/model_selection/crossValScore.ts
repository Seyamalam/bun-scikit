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
  fit(X: Matrix, y: Vector): unknown;
  predict(X: Matrix): Vector;
  score?(X: Matrix, y: Vector): number;
}

export type CrossValSplitter = {
  split(X: Matrix, y?: Vector): FoldIndices[];
};

export interface CrossValScoreOptions {
  cv?: number | CrossValSplitter;
  scoring?: BuiltInScoring | ScoringFn;
}

function isBinaryVector(y: Vector): boolean {
  for (let i = 0; i < y.length; i += 1) {
    const value = y[i];
    if (!(value === 0 || value === 1)) {
      return false;
    }
  }
  return true;
}

function subsetMatrix(X: Matrix, indices: number[]): Matrix {
  const out = new Array(indices.length);
  for (let i = 0; i < indices.length; i += 1) {
    out[i] = X[indices[i]];
  }
  return out;
}

function subsetVector(y: Vector, indices: number[]): Vector {
  const out = new Array(indices.length);
  for (let i = 0; i < indices.length; i += 1) {
    out[i] = y[indices[i]];
  }
  return out;
}

function resolveBuiltInScorer(scoring: BuiltInScoring): ScoringFn {
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
      return meanSquaredError;
    case "neg_mean_squared_error":
      return (yTrue, yPred) => -meanSquaredError(yTrue, yPred);
    default: {
      const exhaustive: never = scoring;
      throw new Error(`Unsupported scoring metric: ${exhaustive}`);
    }
  }
}

function resolveFolds(X: Matrix, y: Vector, cv: number | CrossValSplitter | undefined): FoldIndices[] {
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
    return cv.split(X, y);
  }

  if (isBinaryVector(y)) {
    return new StratifiedKFold({ nSplits: 5, shuffle: false }).split(X, y);
  }
  return new KFold({ nSplits: 5, shuffle: false }).split(X, y);
}

export function crossValScore(
  createEstimator: () => CrossValEstimator,
  X: Matrix,
  y: Vector,
  options: CrossValScoreOptions = {},
): number[] {
  if (typeof createEstimator !== "function") {
    throw new Error("createEstimator must be a function returning a new estimator instance.");
  }

  if (!Array.isArray(X) || X.length === 0) {
    throw new Error("X must be a non-empty matrix.");
  }
  assertFiniteMatrix(X);
  assertVectorLength(y, X.length);
  assertFiniteVector(y);

  const folds = resolveFolds(X, y, options.cv);
  if (folds.length === 0) {
    throw new Error("Cross-validation splitter produced no folds.");
  }

  const explicitScorer =
    typeof options.scoring === "function"
      ? options.scoring
      : options.scoring
        ? resolveBuiltInScorer(options.scoring)
        : null;

  const scores = new Array<number>(folds.length);
  for (let foldIndex = 0; foldIndex < folds.length; foldIndex += 1) {
    const fold = folds[foldIndex];
    if (fold.trainIndices.length === 0 || fold.testIndices.length === 0) {
      throw new Error(`Fold ${foldIndex} must have non-empty train and test indices.`);
    }

    const XTrain = subsetMatrix(X, fold.trainIndices);
    const yTrain = subsetVector(y, fold.trainIndices);
    const XTest = subsetMatrix(X, fold.testIndices);
    const yTest = subsetVector(y, fold.testIndices);

    const estimator = createEstimator();
    estimator.fit(XTrain, yTrain);

    if (explicitScorer) {
      const yPred = estimator.predict(XTest);
      scores[foldIndex] = explicitScorer(yTest, yPred);
      continue;
    }

    if (typeof estimator.score === "function") {
      scores[foldIndex] = estimator.score(XTest, yTest);
      continue;
    }

    throw new Error(
      "Estimator must implement score() when no explicit scoring function is provided.",
    );
  }

  return scores;
}
