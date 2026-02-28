import type { Matrix, Vector } from "../types";
import { r2Score } from "../metrics/regression";
import { fitWithSampleWeight, type FitSampleWeightRequest } from "../utils/fitWithSampleWeight";
import {
  emptyPredictionMatrix,
  extractColumn,
  resolveEstimatorClone,
  setColumn,
  validateMultiOutputInputs,
  validateSampleWeight,
} from "./shared";

interface RegressorLike {
  fit(X: Matrix, y: Vector, sampleWeight?: Vector): unknown;
  predict(X: Matrix): Vector;
}

export class MultiOutputRegressor {
  estimators_: RegressorLike[] = [];
  sampleWeightRequest_ = true;

  private estimatorFactory: (() => RegressorLike) | RegressorLike;
  private nOutputs_: number | null = null;
  private fitted = false;

  constructor(estimatorFactory: (() => RegressorLike) | RegressorLike) {
    this.estimatorFactory = estimatorFactory;
  }

  fit(X: Matrix, Y: Matrix, sampleWeight?: Vector): this {
    validateMultiOutputInputs(X, Y);
    validateSampleWeight(sampleWeight, X.length);

    const nOutputs = Y[0].length;
    const routedSampleWeight = this.sampleWeightRequest_ ? sampleWeight : undefined;
    this.estimators_ = new Array<RegressorLike>(nOutputs);

    for (let outputIndex = 0; outputIndex < nOutputs; outputIndex += 1) {
      const target = extractColumn(Y, outputIndex);
      const estimator = resolveEstimatorClone(this.estimatorFactory);
      fitWithSampleWeight(estimator, X, target, routedSampleWeight);
      this.estimators_[outputIndex] = estimator;
    }

    this.nOutputs_ = nOutputs;
    this.fitted = true;
    return this;
  }

  predict(X: Matrix): Matrix {
    this.assertFitted();
    const predictions = emptyPredictionMatrix(X.length, this.nOutputs_!);
    for (let outputIndex = 0; outputIndex < this.nOutputs_!; outputIndex += 1) {
      const pred = this.estimators_[outputIndex].predict(X);
      setColumn(predictions, outputIndex, pred);
    }
    return predictions;
  }

  score(X: Matrix, Y: Matrix): number {
    validateMultiOutputInputs(X, Y);
    return r2Score(Y, this.predict(X)) as number;
  }

  setFitRequest(request: FitSampleWeightRequest): this {
    if (typeof request.sampleWeight === "boolean") {
      this.sampleWeightRequest_ = request.sampleWeight;
    }
    return this;
  }

  private assertFitted(): void {
    if (!this.fitted || this.nOutputs_ === null || this.estimators_.length === 0) {
      throw new Error("MultiOutputRegressor has not been fitted.");
    }
  }
}
