import type { Matrix, Vector } from "../types";
import { fitWithSampleWeight, type FitSampleWeightRequest } from "../utils/fitWithSampleWeight";
import {
  augmentWithColumns,
  emptyPredictionMatrix,
  exactMatchAccuracy,
  extractColumn,
  randomPermutation,
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

export interface ClassifierChainOptions {
  order?: "random" | number[];
  randomState?: number;
}

function validateOrder(order: number[], nOutputs: number): void {
  if (order.length !== nOutputs) {
    throw new Error(`order must include exactly ${nOutputs} indices.`);
  }
  const seen = new Array<boolean>(nOutputs).fill(false);
  for (let i = 0; i < order.length; i += 1) {
    const index = order[i];
    if (!Number.isInteger(index) || index < 0 || index >= nOutputs) {
      throw new Error(`order contains invalid output index ${index}.`);
    }
    if (seen[index]) {
      throw new Error(`order contains duplicate output index ${index}.`);
    }
    seen[index] = true;
  }
}

function positiveProba(estimator: ClassifierLike, X: Matrix): Vector {
  if (typeof estimator.predictProba !== "function") {
    return estimator.predict(X).map((value) => (value > 0 ? 1 : 0));
  }
  const proba = estimator.predictProba(X);
  return proba.map((row) => {
    if (row.length === 0) {
      return 0;
    }
    return row.length === 1 ? row[0] : row[row.length - 1];
  });
}

export class ClassifierChain {
  estimators_: ClassifierLike[] = [];
  classes_: Vector[] = [];
  order_: number[] = [];
  sampleWeightRequest_ = true;

  private estimatorFactory: (() => ClassifierLike) | ClassifierLike;
  private orderOption: "random" | number[] | undefined;
  private randomState: number;
  private nOutputs_: number | null = null;
  private fitted = false;

  constructor(
    estimatorFactory: (() => ClassifierLike) | ClassifierLike,
    options: ClassifierChainOptions = {},
  ) {
    this.estimatorFactory = estimatorFactory;
    this.orderOption = options.order;
    this.randomState = options.randomState ?? 42;
  }

  fit(X: Matrix, Y: Matrix, sampleWeight?: Vector): this {
    validateMultiOutputInputs(X, Y);
    validateSampleWeight(sampleWeight, X.length);

    const nOutputs = Y[0].length;
    const order =
      this.orderOption === "random"
        ? randomPermutation(nOutputs, this.randomState)
        : this.orderOption
          ? this.orderOption.slice()
          : Array.from({ length: nOutputs }, (_, index) => index);
    validateOrder(order, nOutputs);

    const routedSampleWeight = this.sampleWeightRequest_ ? sampleWeight : undefined;
    this.estimators_ = new Array<ClassifierLike>(nOutputs);
    this.classes_ = new Array<Vector>(nOutputs);
    this.order_ = order;

    for (let position = 0; position < nOutputs; position += 1) {
      const outputIndex = order[position];
      const conditioningOrder = order.slice(0, position);
      const XAugmented = augmentWithColumns(X, Y, conditioningOrder);
      const target = extractColumn(Y, outputIndex);
      const estimator = resolveEstimatorClone(this.estimatorFactory);
      fitWithSampleWeight(estimator, XAugmented, target, routedSampleWeight);
      this.estimators_[position] = estimator;
      this.classes_[position] = estimator.classes_ ? estimator.classes_.slice() : [];
    }

    this.nOutputs_ = nOutputs;
    this.fitted = true;
    return this;
  }

  predict(X: Matrix): Matrix {
    this.assertFitted();
    const predictions = emptyPredictionMatrix(X.length, this.nOutputs_!);
    for (let position = 0; position < this.nOutputs_!; position += 1) {
      const outputIndex = this.order_[position];
      const conditioningOrder = this.order_.slice(0, position);
      const XAugmented = augmentWithColumns(X, predictions, conditioningOrder);
      const pred = this.estimators_[position].predict(XAugmented);
      setColumn(predictions, outputIndex, pred);
    }
    return predictions;
  }

  predictProba(X: Matrix): Matrix {
    this.assertFitted();
    const predictions = emptyPredictionMatrix(X.length, this.nOutputs_!);
    const outputProba = emptyPredictionMatrix(X.length, this.nOutputs_!);
    for (let position = 0; position < this.nOutputs_!; position += 1) {
      const outputIndex = this.order_[position];
      const conditioningOrder = this.order_.slice(0, position);
      const XAugmented = augmentWithColumns(X, predictions, conditioningOrder);
      const pred = this.estimators_[position].predict(XAugmented);
      const proba = positiveProba(this.estimators_[position], XAugmented);
      setColumn(predictions, outputIndex, pred);
      setColumn(outputProba, outputIndex, proba);
    }
    return outputProba;
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
      throw new Error("ClassifierChain has not been fitted.");
    }
  }
}
