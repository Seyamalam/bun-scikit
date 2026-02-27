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

export interface ValidationCurveOptions {
  cv?: number | CrossValSplitter;
  scoring?: BuiltInScoring | ScoringFn;
  groups?: Vector;
  paramName: string;
  paramRange: readonly unknown[];
  sampleWeight?: Vector;
}

export interface ValidationCurveResult {
  paramRange: unknown[];
  trainScores: number[][];
  testScores: number[][];
}

interface SetParamsLike {
  setParams(params: Record<string, unknown>): unknown;
}

function hasSetParams(estimator: CrossValEstimator): estimator is CrossValEstimator & SetParamsLike {
  return typeof (estimator as Partial<SetParamsLike>).setParams === "function";
}

function setEstimatorParam(estimator: CrossValEstimator, paramName: string, value: unknown): void {
  if (hasSetParams(estimator)) {
    estimator.setParams({ [paramName]: value });
    return;
  }

  if (paramName.includes("__")) {
    throw new Error(
      `Estimator does not implement setParams() for nested parameter '${paramName}'.`,
    );
  }

  (estimator as unknown as Record<string, unknown>)[paramName] = value;
}

export function validationCurve(
  createEstimator: () => CrossValEstimator,
  X: Matrix,
  y: Vector,
  options: ValidationCurveOptions,
): ValidationCurveResult {
  if (typeof createEstimator !== "function") {
    throw new Error("createEstimator must be a function returning a new estimator instance.");
  }
  if (typeof options.paramName !== "string" || options.paramName.trim().length === 0) {
    throw new Error("paramName must be a non-empty string.");
  }
  if (!Array.isArray(options.paramRange) || options.paramRange.length === 0) {
    throw new Error("paramRange must be a non-empty array.");
  }

  validateCrossValInputs(X, y, options.groups, options.sampleWeight);
  const folds = resolveFolds(X, y, options.cv, options.groups);
  if (folds.length === 0) {
    throw new Error("Cross-validation splitter produced no folds.");
  }

  const paramRange = Array.from(options.paramRange);
  const trainScores = paramRange.map(() => new Array<number>(folds.length));
  const testScores = paramRange.map(() => new Array<number>(folds.length));

  for (let paramIndex = 0; paramIndex < paramRange.length; paramIndex += 1) {
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
      setEstimatorParam(estimator, options.paramName, paramRange[paramIndex]);
      estimator.fit(XTrain, yTrain, foldSampleWeight);

      trainScores[paramIndex][foldIndex] = evaluateEstimatorScore(
        estimator,
        XTrain,
        yTrain,
        options.scoring,
      );
      testScores[paramIndex][foldIndex] = evaluateEstimatorScore(
        estimator,
        XTest,
        yTest,
        options.scoring,
      );
    }
  }

  return { paramRange, trainScores, testScores };
}
