import type { ClassificationModel, Matrix, Vector } from "../types";
import { accuracyScore } from "../metrics/classification";
import {
  argmax,
  buildLabelIndex,
  uniqueSortedLabels,
} from "../utils/classification";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  validateClassificationInputs,
} from "../utils/validation";

export type RadiusNeighborsClassifierWeights = "uniform" | "distance";

export interface RadiusNeighborsClassifierOptions {
  radius?: number;
  weights?: RadiusNeighborsClassifierWeights;
  outlierLabel?: number | null;
}

function euclideanDistance(a: Vector, b: Vector): number {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

export class RadiusNeighborsClassifier implements ClassificationModel {
  classes_: Vector = [0, 1];

  private radius: number;
  private weights: RadiusNeighborsClassifierWeights;
  private outlierLabel: number | null;
  private XTrain: Matrix | null = null;
  private yTrain: Vector | null = null;
  private labelToIndex: Map<number, number> = new Map<number, number>();

  constructor(options: RadiusNeighborsClassifierOptions = {}) {
    this.radius = options.radius ?? 1;
    this.weights = options.weights ?? "uniform";
    this.outlierLabel = options.outlierLabel ?? null;
    if (!Number.isFinite(this.radius) || this.radius <= 0) {
      throw new Error(`radius must be finite and > 0. Got ${this.radius}.`);
    }
    if (!(this.weights === "uniform" || this.weights === "distance")) {
      throw new Error(`weights must be 'uniform' or 'distance'. Got ${this.weights}.`);
    }
  }

  fit(X: Matrix, y: Vector, sampleWeight?: Vector): this {
    validateClassificationInputs(X, y);
    this.XTrain = X.map((row) => row.slice());
    this.yTrain = y.slice();
    this.classes_ = uniqueSortedLabels(y);
    this.labelToIndex = buildLabelIndex(this.classes_);
    return this;
  }

  predictProba(X: Matrix): Matrix {
    this.assertFitted();
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.XTrain![0].length) {
      throw new Error(`Feature size mismatch. Expected ${this.XTrain![0].length}, got ${X[0].length}.`);
    }
    return X.map((sample) => {
      const counts = new Array<number>(this.classes_.length).fill(0);
      let total = 0;
      for (let i = 0; i < this.XTrain!.length; i += 1) {
        const dist = euclideanDistance(sample, this.XTrain![i]);
        if (dist > this.radius) {
          continue;
        }
        const classIndex = this.labelToIndex.get(this.yTrain![i]);
        if (classIndex === undefined) {
          continue;
        }
        const weight = this.weights === "distance" ? 1 / Math.max(dist, 1e-12) : 1;
        counts[classIndex] += weight;
        total += weight;
      }
      if (total <= 0) {
        return counts;
      }
      return counts.map((value) => value / total);
    });
  }

  predict(X: Matrix): Vector {
    const proba = this.predictProba(X);
    return proba.map((row, index) => {
      let total = 0;
      for (let i = 0; i < row.length; i += 1) {
        total += row[i];
      }
      if (total <= 0 && this.outlierLabel !== null) {
        return this.outlierLabel;
      }
      if (total <= 0) {
        throw new Error(
          `No neighbors found within radius for sample index ${index}. Set outlierLabel to handle outliers.`,
        );
      }
      return this.classes_[argmax(row)];
    });
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return accuracyScore(y, this.predict(X));
  }

  private assertFitted(): void {
    if (!this.XTrain || !this.yTrain) {
      throw new Error("RadiusNeighborsClassifier has not been fitted.");
    }
  }
}
