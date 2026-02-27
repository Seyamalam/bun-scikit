import type { Matrix, Vector } from "../types";
import {
  resolveFolds,
  subsetMatrix,
  subsetVector,
  validateCrossValInputs,
  type CrossValEstimator,
  type CrossValSplitter,
} from "./shared";

export type CrossValPredictMethod = "predict" | "predictProba" | "decisionFunction";

export interface CrossValPredictOptions {
  cv?: number | CrossValSplitter;
  groups?: Vector;
  method?: CrossValPredictMethod;
  sampleWeight?: Vector;
}

interface ProbaEstimator extends CrossValEstimator {
  predictProba(X: Matrix): Matrix;
}

interface DecisionEstimator extends CrossValEstimator {
  decisionFunction(X: Matrix): Vector | Matrix;
}

function hasPredictProba(estimator: CrossValEstimator): estimator is ProbaEstimator {
  return typeof (estimator as Partial<ProbaEstimator>).predictProba === "function";
}

function hasDecisionFunction(estimator: CrossValEstimator): estimator is DecisionEstimator {
  return typeof (estimator as Partial<DecisionEstimator>).decisionFunction === "function";
}

function isVectorPrediction(value: unknown): value is Vector {
  return Array.isArray(value) && (value.length === 0 || typeof value[0] === "number");
}

function isMatrixPrediction(value: unknown): value is Matrix {
  return Array.isArray(value) && (value.length === 0 || Array.isArray(value[0]));
}

export function crossValPredict(
  createEstimator: () => CrossValEstimator,
  X: Matrix,
  y: Vector,
  options: CrossValPredictOptions = {},
): Vector | Matrix {
  if (typeof createEstimator !== "function") {
    throw new Error("createEstimator must be a function returning a new estimator instance.");
  }

  validateCrossValInputs(X, y, options.groups, options.sampleWeight);
  const method = options.method ?? "predict";
  const folds = resolveFolds(X, y, options.cv, options.groups);
  if (folds.length === 0) {
    throw new Error("Cross-validation splitter produced no folds.");
  }

  const seen = new Uint16Array(X.length);
  const vectorOut = new Array<number>(X.length);
  const matrixOut = new Array<number[]>(X.length);
  let outputShape: "vector" | "matrix" | null = null;

  for (let foldIndex = 0; foldIndex < folds.length; foldIndex += 1) {
    const fold = folds[foldIndex];
    if (fold.trainIndices.length === 0 || fold.testIndices.length === 0) {
      throw new Error(`Fold ${foldIndex} must have non-empty train and test indices.`);
    }

    const XTrain = subsetMatrix(X, fold.trainIndices);
    const yTrain = subsetVector(y, fold.trainIndices);
    const XTest = subsetMatrix(X, fold.testIndices);
    const foldSampleWeight = options.sampleWeight
      ? subsetVector(options.sampleWeight, fold.trainIndices)
      : undefined;

    const estimator = createEstimator();
    estimator.fit(XTrain, yTrain, foldSampleWeight);

    let foldPred: Vector | Matrix;
    if (method === "predict") {
      foldPred = estimator.predict(XTest);
    } else if (method === "predictProba") {
      if (!hasPredictProba(estimator)) {
        throw new Error("crossValPredict(method='predictProba') requires estimator.predictProba().");
      }
      foldPred = estimator.predictProba(XTest);
    } else {
      if (!hasDecisionFunction(estimator)) {
        throw new Error(
          "crossValPredict(method='decisionFunction') requires estimator.decisionFunction().",
        );
      }
      foldPred = estimator.decisionFunction(XTest);
    }

    if (isVectorPrediction(foldPred)) {
      if (foldPred.length !== fold.testIndices.length) {
        throw new Error(
          `Fold ${foldIndex} returned ${foldPred.length} predictions for ${fold.testIndices.length} test rows.`,
        );
      }
      if (outputShape === "matrix") {
        throw new Error("crossValPredict received mixed vector and matrix outputs across folds.");
      }
      outputShape = "vector";
      for (let i = 0; i < fold.testIndices.length; i += 1) {
        const originalIndex = fold.testIndices[i];
        if (seen[originalIndex] !== 0) {
          throw new Error(
            `crossValPredict requires non-overlapping test folds. Index ${originalIndex} appeared multiple times.`,
          );
        }
        seen[originalIndex] = 1;
        vectorOut[originalIndex] = foldPred[i];
      }
      continue;
    }

    if (isMatrixPrediction(foldPred)) {
      if (foldPred.length !== fold.testIndices.length) {
        throw new Error(
          `Fold ${foldIndex} returned ${foldPred.length} predictions for ${fold.testIndices.length} test rows.`,
        );
      }
      if (outputShape === "vector") {
        throw new Error("crossValPredict received mixed vector and matrix outputs across folds.");
      }
      outputShape = "matrix";
      for (let i = 0; i < fold.testIndices.length; i += 1) {
        const originalIndex = fold.testIndices[i];
        if (seen[originalIndex] !== 0) {
          throw new Error(
            `crossValPredict requires non-overlapping test folds. Index ${originalIndex} appeared multiple times.`,
          );
        }
        seen[originalIndex] = 1;
        matrixOut[originalIndex] = foldPred[i];
      }
      continue;
    }

    throw new Error("Estimator prediction output must be a vector or matrix.");
  }

  for (let i = 0; i < seen.length; i += 1) {
    if (seen[i] !== 1) {
      throw new Error(
        `crossValPredict requires that every sample appears in test exactly once. Missing index ${i}.`,
      );
    }
  }

  if (outputShape === "matrix") {
    return matrixOut;
  }
  return vectorOut;
}
