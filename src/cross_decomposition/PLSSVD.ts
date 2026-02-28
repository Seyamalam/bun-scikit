import type { Matrix, Vector } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";
import { multiplyMatrices, transpose } from "../utils/linalg";
import {
  centerAndScale,
  topSingularVectors,
  toTargetMatrix,
  validateCrossDecompositionInputs,
} from "./shared";

export interface PLSSVDOptions {
  nComponents?: number;
  scale?: boolean;
  maxIter?: number;
  tolerance?: number;
}

export class PLSSVD {
  xWeights_: Matrix | null = null;
  yWeights_: Matrix | null = null;
  xScores_: Matrix | null = null;
  yScores_: Matrix | null = null;
  singularValues_: Vector | null = null;
  xMean_: Vector | null = null;
  yMean_: Vector | null = null;
  xStd_: Vector | null = null;
  yStd_: Vector | null = null;
  nFeaturesIn_: number | null = null;
  nTargetsIn_: number | null = null;

  protected nComponents?: number;
  protected scale: boolean;
  protected maxIter: number;
  protected tolerance: number;
  protected targetIsVector = false;
  protected fitted = false;

  constructor(options: PLSSVDOptions = {}) {
    this.nComponents = options.nComponents;
    this.scale = options.scale ?? true;
    this.maxIter = options.maxIter ?? 500;
    this.tolerance = options.tolerance ?? 1e-8;

    if (typeof this.scale !== "boolean") {
      throw new Error(`scale must be a boolean. Got ${this.scale as unknown as string}.`);
    }
    if (this.nComponents !== undefined && (!Number.isInteger(this.nComponents) || this.nComponents < 1)) {
      throw new Error(`nComponents must be an integer >= 1 when provided. Got ${this.nComponents}.`);
    }
    if (!Number.isInteger(this.maxIter) || this.maxIter < 1) {
      throw new Error(`maxIter must be an integer >= 1. Got ${this.maxIter}.`);
    }
    if (!Number.isFinite(this.tolerance) || this.tolerance <= 0) {
      throw new Error(`tolerance must be finite and > 0. Got ${this.tolerance}.`);
    }
  }

  fit(X: Matrix, Y: Matrix | Vector): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    const target = toTargetMatrix(Y);
    validateCrossDecompositionInputs(X, target.Y);
    this.targetIsVector = target.targetIsVector;

    const xProcessed = centerAndScale(X, this.scale);
    const yProcessed = centerAndScale(target.Y, this.scale);
    const xCrossY = multiplyMatrices(transpose(xProcessed.transformed), yProcessed.transformed);

    const nComponents = Math.min(
      this.nComponents ?? Math.min(X[0].length, target.Y[0].length),
      X[0].length,
      target.Y[0].length,
    );
    const svd = topSingularVectors(xCrossY, nComponents, this.maxIter, this.tolerance);
    const xScores = multiplyMatrices(xProcessed.transformed, svd.left);
    const yScores = multiplyMatrices(yProcessed.transformed, svd.right);

    this.xWeights_ = svd.left;
    this.yWeights_ = svd.right;
    this.xScores_ = xScores;
    this.yScores_ = yScores;
    this.singularValues_ = svd.singularValues;
    this.xMean_ = xProcessed.mean;
    this.yMean_ = yProcessed.mean;
    this.xStd_ = xProcessed.scale;
    this.yStd_ = yProcessed.scale;
    this.nFeaturesIn_ = X[0].length;
    this.nTargetsIn_ = target.Y[0].length;
    this.fitted = true;
    return this;
  }

  transform(X: Matrix): Matrix {
    this.assertFitted();
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }
    const normalized: Matrix = new Array(X.length);
    for (let i = 0; i < X.length; i += 1) {
      const row = new Array<number>(this.nFeaturesIn_!);
      for (let j = 0; j < this.nFeaturesIn_!; j += 1) {
        row[j] = (X[i][j] - this.xMean_![j]) / this.xStd_![j];
      }
      normalized[i] = row;
    }
    return multiplyMatrices(normalized, this.xWeights_!);
  }

  fitTransform(X: Matrix, Y: Matrix | Vector): Matrix {
    return this.fit(X, Y).transform(X);
  }

  protected assertFitted(): void {
    if (
      !this.fitted ||
      !this.xWeights_ ||
      !this.yWeights_ ||
      !this.xMean_ ||
      !this.yMean_ ||
      !this.xStd_ ||
      !this.yStd_ ||
      this.nFeaturesIn_ === null ||
      this.nTargetsIn_ === null
    ) {
      throw new Error("PLSSVD has not been fitted.");
    }
  }
}
