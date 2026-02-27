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

export interface CrossValScoreOptions {
  cv?: number | CrossValSplitter;
  scoring?: BuiltInScoring | ScoringFn;
  groups?: Vector;
  sampleWeight?: Vector;
}

export type { BuiltInScoring, CrossValEstimator, CrossValSplitter, ScoringFn } from "./shared";

export function crossValScore(
  createEstimator: () => CrossValEstimator,
  X: Matrix,
  y: Vector,
  options: CrossValScoreOptions = {},
): number[] {
  if (typeof createEstimator !== "function") {
    throw new Error("createEstimator must be a function returning a new estimator instance.");
  }

  validateCrossValInputs(X, y, options.groups, options.sampleWeight);

  const folds = resolveFolds(X, y, options.cv, options.groups);
  if (folds.length === 0) {
    throw new Error("Cross-validation splitter produced no folds.");
  }

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
    const foldSampleWeight = options.sampleWeight
      ? subsetVector(options.sampleWeight, fold.trainIndices)
      : undefined;

    const estimator = createEstimator();
    estimator.fit(XTrain, yTrain, foldSampleWeight);
    scores[foldIndex] = evaluateEstimatorScore(estimator, XTest, yTest, options.scoring);
  }

  return scores;
}
