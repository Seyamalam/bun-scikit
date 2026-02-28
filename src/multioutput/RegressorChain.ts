import type { Matrix, Vector } from "../types";
import { r2Score } from "../metrics/regression";
import { fitWithSampleWeight, type FitSampleWeightRequest } from "../utils/fitWithSampleWeight";
import {
  augmentWithColumns,
  emptyPredictionMatrix,
  extractColumn,
  randomPermutation,
  resolveEstimatorClone,
  setColumn,
  validateMultiOutputInputs,
  validateSampleWeight,
} from "./shared";

interface RegressorLike {
  fit(X: Matrix, y: Vector, sampleWeight?: Vector): unknown;
  predict(X: Matrix): Vector;
}

export interface RegressorChainOptions {
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

export class RegressorChain {
  estimators_: RegressorLike[] = [];
  order_: number[] = [];
  sampleWeightRequest_ = true;

  private estimatorFactory: (() => RegressorLike) | RegressorLike;
  private orderOption: "random" | number[] | undefined;
  private randomState: number;
  private nOutputs_: number | null = null;
  private fitted = false;

  constructor(
    estimatorFactory: (() => RegressorLike) | RegressorLike,
    options: RegressorChainOptions = {},
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
    this.estimators_ = new Array<RegressorLike>(nOutputs);
    this.order_ = order;

    for (let position = 0; position < nOutputs; position += 1) {
      const outputIndex = order[position];
      const conditioningOrder = order.slice(0, position);
      const XAugmented = augmentWithColumns(X, Y, conditioningOrder);
      const target = extractColumn(Y, outputIndex);
      const estimator = resolveEstimatorClone(this.estimatorFactory);
      fitWithSampleWeight(estimator, XAugmented, target, routedSampleWeight);
      this.estimators_[position] = estimator;
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
      throw new Error("RegressorChain has not been fitted.");
    }
  }
}
