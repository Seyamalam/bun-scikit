import type { ClassificationModel, Matrix, Vector } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  validateClassificationInputs,
} from "../utils/validation";
import { accuracyScore } from "../metrics/classification";

export interface KNeighborsClassifierOptions {
  nNeighbors?: number;
}

function squaredEuclideanDistance(a: Vector, b: Vector): number {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

export class KNeighborsClassifier implements ClassificationModel {
  classes_: Vector = [0, 1];
  private readonly nNeighbors: number;
  private XTrain: Matrix | null = null;
  private yTrain: Vector | null = null;

  constructor(options: KNeighborsClassifierOptions = {}) {
    const nNeighbors = options.nNeighbors ?? 5;
    if (!Number.isInteger(nNeighbors) || nNeighbors < 1) {
      throw new Error(`nNeighbors must be a positive integer. Got ${nNeighbors}.`);
    }
    this.nNeighbors = nNeighbors;
  }

  fit(X: Matrix, y: Vector): this {
    validateClassificationInputs(X, y);
    if (this.nNeighbors > X.length) {
      throw new Error(
        `nNeighbors (${this.nNeighbors}) cannot exceed training size (${X.length}).`,
      );
    }

    this.XTrain = X.map((row) => [...row]);
    this.yTrain = [...y];
    return this;
  }

  predict(X: Matrix): Vector {
    if (!this.XTrain || !this.yTrain) {
      throw new Error("KNeighborsClassifier has not been fitted.");
    }

    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.XTrain[0].length) {
      throw new Error(
        `Feature size mismatch. Expected ${this.XTrain[0].length}, got ${X[0].length}.`,
      );
    }

    return X.map((sample) => {
      const distances = this.XTrain!.map((row, idx) => ({
        distance: squaredEuclideanDistance(sample, row),
        label: this.yTrain![idx],
      }));
      distances.sort((a, b) => a.distance - b.distance);

      let positiveVotes = 0;
      for (let i = 0; i < this.nNeighbors; i += 1) {
        if (distances[i].label === 1) {
          positiveVotes += 1;
        }
      }

      return positiveVotes * 2 >= this.nNeighbors ? 1 : 0;
    });
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return accuracyScore(y, this.predict(X));
  }
}
