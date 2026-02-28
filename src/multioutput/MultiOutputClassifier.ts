import type { Matrix, Vector } from "../types";
import { fitWithSampleWeight, type FitSampleWeightRequest } from "../utils/fitWithSampleWeight";
import {
  emptyPredictionMatrix,
  exactMatchAccuracy,
  extractColumn,
  resolveEstimatorClone,
  setColumn,
  validateMultiOutputInputs,
  validateSampleWeight,
} from "./shared";

interface ClassifierLike {
  classes_?: Vector | null;
  fit(X: Matrix, y: Vector, sampleWeight?: Vector): unknown;
  predict(X: Matrix): Vector;
  predictProba?: (X: Matrix) => Matrix;
}

export class MultiOutputClassifier {
  estimators_: ClassifierLike[] = [];
  classes_: Vector[] = [];
  sampleWeightRequest_ = true;

  private estimatorFactory: (() => ClassifierLike) | ClassifierLike;
  private nOutputs_: number | null = null;
  private fitted = false;

  constructor(estimatorFactory: (() => ClassifierLike) | ClassifierLike) {
    this.estimatorFactory = estimatorFactory;
  }

  fit(X: Matrix, Y: Matrix, sampleWeight?: Vector): this {
    validateMultiOutputInputs(X, Y);
    validateSampleWeight(sampleWeight, X.length);

    const nOutputs = Y[0].length;
    this.estimators_ = new Array<ClassifierLike>(nOutputs);
    this.classes_ = new Array<Vector>(nOutputs);
    const routedSampleWeight = this.sampleWeightRequest_ ? sampleWeight : undefined;

    for (let outputIndex = 0; outputIndex < nOutputs; outputIndex += 1) {
      const target = extractColumn(Y, outputIndex);
      const estimator = resolveEstimatorClone(this.estimatorFactory);
      fitWithSampleWeight(estimator, X, target, routedSampleWeight);
      this.estimators_[outputIndex] = estimator;
      this.classes_[outputIndex] = estimator.classes_ ? estimator.classes_.slice() : [];
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

  predictProba(X: Matrix): Matrix[] {
    this.assertFitted();
    const out: Matrix[] = new Array(this.nOutputs_!);
    for (let outputIndex = 0; outputIndex < this.nOutputs_!; outputIndex += 1) {
      const estimator = this.estimators_[outputIndex];
      if (typeof estimator.predictProba !== "function") {
        throw new Error(
          `Estimator at output ${outputIndex} does not implement predictProba.`,
        );
      }
      out[outputIndex] = estimator.predictProba(X);
    }
    return out;
  }

  score(X: Matrix, Y: Matrix): number {
    validateMultiOutputInputs(X, Y);
    return exactMatchAccuracy(Y, this.predict(X));
  }

  setFitRequest(request: FitSampleWeightRequest): this {
    if (typeof request.sampleWeight === "boolean") {
      this.sampleWeightRequest_ = request.sampleWeight;
    }
    return this;
  }

  private assertFitted(): void {
    if (!this.fitted || this.nOutputs_ === null || this.estimators_.length === 0) {
      throw new Error("MultiOutputClassifier has not been fitted.");
    }
  }
}
