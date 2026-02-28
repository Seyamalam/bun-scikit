import type { Matrix, Vector } from "../types";
import { KFold } from "../model_selection/KFold";
import { meanSquaredError } from "./sharedRegularized";

export interface CrossValConfig {
  cv: number;
  randomState?: number;
}

export function crossValidatedMse(
  factory: () => { fit: (X: Matrix, y: Vector) => unknown; predict: (X: Matrix) => Vector },
  X: Matrix,
  y: Vector,
  config: CrossValConfig,
): number {
  const splitter = new KFold({
    nSplits: config.cv,
    shuffle: true,
    randomState: config.randomState ?? 42,
  });
  const folds = splitter.split(X, y);
  let total = 0;
  for (let i = 0; i < folds.length; i += 1) {
    const trainIndices = folds[i].trainIndices;
    const testIndices = folds[i].testIndices;

    const XTrain = trainIndices.map((idx) => X[idx]);
    const yTrain = trainIndices.map((idx) => y[idx]);
    const XTest = testIndices.map((idx) => X[idx]);
    const yTest = testIndices.map((idx) => y[idx]);

    const estimator = factory();
    estimator.fit(XTrain, yTrain);
    const pred = estimator.predict(XTest);
    total += meanSquaredError(yTest, pred);
  }
  return total / folds.length;
}
