import type { Matrix, Vector } from "../types";
import { accuracyScore, f1Score, precisionScore, recallScore } from "../metrics/classification";
import { meanSquaredError, r2Score } from "../metrics/regression";
import { assertFiniteMatrix, assertFiniteVector, assertVectorLength } from "../utils/validation";
import { KFold, type FoldIndices } from "./KFold";
import { StratifiedKFold } from "./StratifiedKFold";
import type {
  BuiltInScoring,
  CrossValEstimator,
  CrossValSplitter,
  ScoringFn,
} from "./crossValScore";

export interface CrossValidateOptions {
  cv?: number | CrossValSplitter;
  scoring?: BuiltInScoring | ScoringFn;
  groups?: Vector;
  returnTrainScore?: boolean;
  returnEstimator?: boolean;
}

export interface CrossValidateResult {
  fitTime: number[];
  scoreTime: number[];
  testScore: number[];
  trainScore?: number[];
  estimators?: CrossValEstimator[];
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

function nowMs(): number {
  if (typeof performance !== "undefined" && typeof performance.now === "function") {
    return performance.now();
  }
  return Date.now();
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

function resolveFolds(
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

export function crossValidate(
  createEstimator: () => CrossValEstimator,
  X: Matrix,
  y: Vector,
  options: CrossValidateOptions = {},
): CrossValidateResult {
  if (typeof createEstimator !== "function") {
    throw new Error("createEstimator must be a function returning a new estimator instance.");
  }
  if (!Array.isArray(X) || X.length === 0) {
    throw new Error("X must be a non-empty matrix.");
  }

  assertFiniteMatrix(X);
  assertVectorLength(y, X.length);
  assertFiniteVector(y);
  if (options.groups) {
    assertVectorLength(options.groups, X.length);
    assertFiniteVector(options.groups);
  }

  const scorer =
    typeof options.scoring === "function"
      ? options.scoring
      : options.scoring
        ? resolveBuiltInScorer(options.scoring)
        : null;
  const folds = resolveFolds(X, y, options.cv, options.groups);
  if (folds.length === 0) {
    throw new Error("Cross-validation splitter produced no folds.");
  }

  const fitTime = new Array<number>(folds.length);
  const scoreTime = new Array<number>(folds.length);
  const testScore = new Array<number>(folds.length);
  const trainScore = options.returnTrainScore ? new Array<number>(folds.length) : undefined;
  const estimators = options.returnEstimator ? new Array<CrossValEstimator>(folds.length) : undefined;

  for (let foldIndex = 0; foldIndex < folds.length; foldIndex += 1) {
    const fold = folds[foldIndex];
    const XTrain = subsetMatrix(X, fold.trainIndices);
    const yTrain = subsetVector(y, fold.trainIndices);
    const XTest = subsetMatrix(X, fold.testIndices);
    const yTest = subsetVector(y, fold.testIndices);

    const estimator = createEstimator();
    const fitStart = nowMs();
    estimator.fit(XTrain, yTrain);
    fitTime[foldIndex] = nowMs() - fitStart;

    const scoreStart = nowMs();
    if (scorer) {
      const yPred = estimator.predict(XTest);
      testScore[foldIndex] = scorer(yTest, yPred);
      if (trainScore) {
        const yTrainPred = estimator.predict(XTrain);
        trainScore[foldIndex] = scorer(yTrain, yTrainPred);
      }
    } else if (typeof estimator.score === "function") {
      testScore[foldIndex] = estimator.score(XTest, yTest);
      if (trainScore) {
        trainScore[foldIndex] = estimator.score(XTrain, yTrain);
      }
    } else {
      throw new Error(
        "Estimator must implement score() when no explicit scoring function is provided.",
      );
    }
    scoreTime[foldIndex] = nowMs() - scoreStart;

    if (estimators) {
      estimators[foldIndex] = estimator;
    }
  }

  return {
    fitTime,
    scoreTime,
    testScore,
    ...(trainScore ? { trainScore } : {}),
    ...(estimators ? { estimators } : {}),
  };
}
