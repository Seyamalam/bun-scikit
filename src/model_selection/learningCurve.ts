import type { Matrix, Vector } from "../types";
import {
  evaluateEstimatorScore,
  resolveFolds,
  shuffleInPlace,
  subsetMatrix,
  subsetVector,
  validateCrossValInputs,
  type BuiltInScoring,
  type CrossValEstimator,
  type CrossValSplitter,
  type ScoringFn,
} from "./shared";

export interface LearningCurveOptions {
  cv?: number | CrossValSplitter;
  scoring?: BuiltInScoring | ScoringFn;
  groups?: Vector;
  trainSizes?: number[];
  shuffle?: boolean;
  randomState?: number;
  sampleWeight?: Vector;
}

export interface LearningCurveResult {
  trainSizes: number[];
  trainScores: number[][];
  testScores: number[][];
}

const DEFAULT_TRAIN_SIZES = [0.1, 0.325, 0.55, 0.775, 1.0];

function resolveTrainSizes(raw: number[] | undefined, minTrainSize: number): number[] {
  const requested = raw && raw.length > 0 ? raw : DEFAULT_TRAIN_SIZES;
  const resolved = new Set<number>();
  for (let i = 0; i < requested.length; i += 1) {
    const value = requested[i];
    if (!Number.isFinite(value)) {
      throw new Error(`trainSizes values must be finite. Got ${value}.`);
    }

    if (value > 0 && value <= 1) {
      resolved.add(Math.min(minTrainSize, Math.max(1, Math.ceil(minTrainSize * value))));
      continue;
    }

    if (Number.isInteger(value) && value >= 1) {
      if (value > minTrainSize) {
        throw new Error(
          `Absolute trainSizes value ${value} exceeds smallest train fold size ${minTrainSize}.`,
        );
      }
      resolved.add(value);
      continue;
    }

    throw new Error(
      `trainSizes values must be floats in (0, 1] or integers in [1, ${minTrainSize}]. Got ${value}.`,
    );
  }
  return Array.from(resolved).sort((a, b) => a - b);
}

export function learningCurve(
  createEstimator: () => CrossValEstimator,
  X: Matrix,
  y: Vector,
  options: LearningCurveOptions = {},
): LearningCurveResult {
  if (typeof createEstimator !== "function") {
    throw new Error("createEstimator must be a function returning a new estimator instance.");
  }

  validateCrossValInputs(X, y, options.groups, options.sampleWeight);
  const folds = resolveFolds(X, y, options.cv, options.groups);
  if (folds.length === 0) {
    throw new Error("Cross-validation splitter produced no folds.");
  }

  const minTrainSize = Math.min(...folds.map((fold) => fold.trainIndices.length));
  if (!Number.isFinite(minTrainSize) || minTrainSize < 1) {
    throw new Error("Learning curve requires non-empty training folds.");
  }
  const trainSizes = resolveTrainSizes(options.trainSizes, minTrainSize);

  const trainScores = trainSizes.map(() => new Array<number>(folds.length));
  const testScores = trainSizes.map(() => new Array<number>(folds.length));
  const shuffle = options.shuffle ?? false;
  const randomState = options.randomState ?? 42;

  for (let foldIndex = 0; foldIndex < folds.length; foldIndex += 1) {
    const fold = folds[foldIndex];
    const orderedTrainIndices = fold.trainIndices.slice();
    if (shuffle) {
      shuffleInPlace(orderedTrainIndices, randomState + foldIndex);
    }

    const XTest = subsetMatrix(X, fold.testIndices);
    const yTest = subsetVector(y, fold.testIndices);

    for (let trainSizeIndex = 0; trainSizeIndex < trainSizes.length; trainSizeIndex += 1) {
      const trainSize = trainSizes[trainSizeIndex];
      const trainIndices = orderedTrainIndices.slice(0, trainSize);
      const XTrain = subsetMatrix(X, trainIndices);
      const yTrain = subsetVector(y, trainIndices);
      const foldSampleWeight = options.sampleWeight
        ? subsetVector(options.sampleWeight, trainIndices)
        : undefined;

      const estimator = createEstimator();
      estimator.fit(XTrain, yTrain, foldSampleWeight);
      trainScores[trainSizeIndex][foldIndex] = evaluateEstimatorScore(
        estimator,
        XTrain,
        yTrain,
        options.scoring,
      );
      testScores[trainSizeIndex][foldIndex] = evaluateEstimatorScore(
        estimator,
        XTest,
        yTest,
        options.scoring,
      );
    }
  }

  return { trainSizes, trainScores, testScores };
}
