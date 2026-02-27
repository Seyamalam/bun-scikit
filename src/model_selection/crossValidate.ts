import type { Matrix, Vector } from "../types";
import {
  evaluateEstimatorScore,
  resolveFolds,
  subsetMatrix,
  subsetVector,
  validateCrossValInputs,
  type BuiltInScoring,
  type CrossValEstimator,
  type CrossValSplitter,
  type ScoringFn,
} from "./shared";

export interface CrossValidateOptions {
  cv?: number | CrossValSplitter;
  scoring?: BuiltInScoring | ScoringFn;
  groups?: Vector;
  returnTrainScore?: boolean;
  returnEstimator?: boolean;
  sampleWeight?: Vector;
}

export interface CrossValidateResult {
  fitTime: number[];
  scoreTime: number[];
  testScore: number[];
  trainScore?: number[];
  estimators?: CrossValEstimator[];
}

function nowMs(): number {
  if (typeof performance !== "undefined" && typeof performance.now === "function") {
    return performance.now();
  }
  return Date.now();
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
  validateCrossValInputs(X, y, options.groups, options.sampleWeight);

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
    const foldSampleWeight = options.sampleWeight
      ? subsetVector(options.sampleWeight, fold.trainIndices)
      : undefined;

    const estimator = createEstimator();
    const fitStart = nowMs();
    estimator.fit(XTrain, yTrain, foldSampleWeight);
    fitTime[foldIndex] = nowMs() - fitStart;

    const scoreStart = nowMs();
    testScore[foldIndex] = evaluateEstimatorScore(estimator, XTest, yTest, options.scoring);
    if (trainScore) {
      trainScore[foldIndex] = evaluateEstimatorScore(estimator, XTrain, yTrain, options.scoring);
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
