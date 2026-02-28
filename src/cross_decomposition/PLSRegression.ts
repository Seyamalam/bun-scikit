import type { Matrix, Vector } from "../types";
import { inverseMatrix, multiplyMatrices, transpose } from "../utils/linalg";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";
import {
  centerAndScale,
  computeRotations,
  copyMatrix,
  crossVector,
  deflateByOuter,
  denormalizeWithMeanAndScale,
  getColumn,
  matVecDot,
  normalizeWithMeanAndScale,
  setColumn,
  squaredNorm,
  toTargetMatrix,
  topSingularVectors,
  trimColumns,
  validateCrossDecompositionInputs,
} from "./shared";

export interface PLSRegressionOptions {
  nComponents?: number;
  scale?: boolean;
  maxIter?: number;
  tolerance?: number;
}

type PLSDeflationMode = "regression" | "canonical";
type PredictResult = Matrix | Vector;
type TransformResult = Matrix | [Matrix, Matrix];

export class PLSRegression {
  xWeights_: Matrix | null = null;
  yWeights_: Matrix | null = null;
  xLoadings_: Matrix | null = null;
  yLoadings_: Matrix | null = null;
  xScores_: Matrix | null = null;
  yScores_: Matrix | null = null;
  xRotations_: Matrix | null = null;
  yRotations_: Matrix | null = null;
  coef_: Matrix | null = null;
  intercept_: Vector | null = null;
  nIter_: Vector | null = null;
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
  protected deflationMode: PLSDeflationMode = "regression";
  protected targetIsVector = false;
  protected fitted = false;

  constructor(options: PLSRegressionOptions = {}) {
    this.nComponents = options.nComponents;
    this.scale = options.scale ?? true;
    this.maxIter = options.maxIter ?? 500;
    this.tolerance = options.tolerance ?? 1e-8;
    this.validateOptions();
  }

  protected validateOptions(): void {
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

    const nSamples = X.length;
    const nFeatures = X[0].length;
    const nTargets = target.Y[0].length;
    const maxComponents = Math.min(nSamples, nFeatures, nTargets);
    const requestedComponents = this.nComponents ?? maxComponents;
    const nComponents = Math.min(requestedComponents, maxComponents);

    const xWeights = Array.from({ length: nFeatures }, () => new Array<number>(nComponents).fill(0));
    const yWeights = Array.from({ length: nTargets }, () => new Array<number>(nComponents).fill(0));
    const xLoadings = Array.from({ length: nFeatures }, () => new Array<number>(nComponents).fill(0));
    const yLoadings = Array.from({ length: nTargets }, () => new Array<number>(nComponents).fill(0));
    const xScores = Array.from({ length: nSamples }, () => new Array<number>(nComponents).fill(0));
    const yScores = Array.from({ length: nSamples }, () => new Array<number>(nComponents).fill(0));
    const nIter = new Array<number>(nComponents).fill(1);

    const xResidual = copyMatrix(xProcessed.transformed);
    const yResidual = copyMatrix(yProcessed.transformed);

    let fittedComponents = 0;
    for (let component = 0; component < nComponents; component += 1) {
      const cross = multiplyMatrices(transpose(xResidual), yResidual);
      const svd = topSingularVectors(cross, 1, this.maxIter, this.tolerance);
      const xWeight = getColumn(svd.left, 0);
      const yWeight = getColumn(svd.right, 0);

      const xScore = matVecDot(xResidual, xWeight);
      const yScore = matVecDot(yResidual, yWeight);
      const xScoreNorm = squaredNorm(xScore);
      const yScoreNorm = squaredNorm(yScore);
      if (xScoreNorm <= 1e-14 || yScoreNorm <= 1e-14) {
        break;
      }

      const xLoading = crossVector(xResidual, xScore).map((value) => value / xScoreNorm);
      const yLoading =
        this.deflationMode === "canonical"
          ? crossVector(yResidual, yScore).map((value) => value / yScoreNorm)
          : crossVector(yResidual, xScore).map((value) => value / xScoreNorm);

      setColumn(xWeights, component, xWeight);
      setColumn(yWeights, component, yWeight);
      setColumn(xLoadings, component, xLoading);
      setColumn(yLoadings, component, yLoading);
      setColumn(xScores, component, xScore);
      setColumn(yScores, component, yScore);

      deflateByOuter(xResidual, xScore, xLoading);
      if (this.deflationMode === "canonical") {
        deflateByOuter(yResidual, yScore, yLoading);
      } else {
        deflateByOuter(yResidual, xScore, yLoading);
      }
      fittedComponents += 1;
    }

    if (fittedComponents === 0) {
      throw new Error("PLSRegression could not extract any latent component from the input data.");
    }

    const xWeightsTrimmed = trimColumns(xWeights, fittedComponents);
    const yWeightsTrimmed = trimColumns(yWeights, fittedComponents);
    const xLoadingsTrimmed = trimColumns(xLoadings, fittedComponents);
    const yLoadingsTrimmed = trimColumns(yLoadings, fittedComponents);
    const xScoresTrimmed = trimColumns(xScores, fittedComponents);
    const yScoresTrimmed = trimColumns(yScores, fittedComponents);

    const xRotations = computeRotations(xWeightsTrimmed, xLoadingsTrimmed);
    const yRotations = computeRotations(yWeightsTrimmed, yLoadingsTrimmed);
    const xScoreReg = multiplyMatrices(xProcessed.transformed, xRotations);

    const scoreXtX = multiplyMatrices(transpose(xScoreReg), xScoreReg);
    const scoreXtY = multiplyMatrices(transpose(xScoreReg), yProcessed.transformed);
    const ridge = 1e-8;
    const scoreXtXReg = scoreXtX.map((row, i) =>
      row.map((value, j) => (i === j ? value + ridge : value)),
    );
    const scoreCoef = multiplyMatrices(inverseMatrix(scoreXtXReg), scoreXtY);
    const coefScaled = multiplyMatrices(xRotations, scoreCoef);

    const coefTargetByFeature: Matrix = Array.from({ length: nTargets }, () =>
      new Array<number>(nFeatures).fill(0),
    );
    const intercept = new Array<number>(nTargets).fill(0);
    for (let targetIndex = 0; targetIndex < nTargets; targetIndex += 1) {
      let bias = yProcessed.mean[targetIndex];
      for (let featureIndex = 0; featureIndex < nFeatures; featureIndex += 1) {
        const rawCoef =
          (coefScaled[featureIndex][targetIndex] * yProcessed.scale[targetIndex]) /
          xProcessed.scale[featureIndex];
        coefTargetByFeature[targetIndex][featureIndex] = rawCoef;
        bias -= rawCoef * xProcessed.mean[featureIndex];
      }
      intercept[targetIndex] = bias;
    }

    this.xWeights_ = xWeightsTrimmed;
    this.yWeights_ = yWeightsTrimmed;
    this.xLoadings_ = xLoadingsTrimmed;
    this.yLoadings_ = yLoadingsTrimmed;
    this.xScores_ = xScoresTrimmed;
    this.yScores_ = yScoresTrimmed;
    this.xRotations_ = xRotations;
    this.yRotations_ = yRotations;
    this.coef_ = coefTargetByFeature;
    this.intercept_ = intercept;
    this.nIter_ = nIter.slice(0, fittedComponents);
    this.xMean_ = xProcessed.mean;
    this.yMean_ = yProcessed.mean;
    this.xStd_ = xProcessed.scale;
    this.yStd_ = yProcessed.scale;
    this.nFeaturesIn_ = nFeatures;
    this.nTargetsIn_ = nTargets;
    this.fitted = true;
    return this;
  }

  predict(X: Matrix): PredictResult {
    this.assertFitted();
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }

    const predictions: Matrix = new Array(X.length);
    for (let i = 0; i < X.length; i += 1) {
      const row = new Array<number>(this.nTargetsIn_!);
      for (let targetIndex = 0; targetIndex < this.nTargetsIn_!; targetIndex += 1) {
        let value = this.intercept_![targetIndex];
        for (let featureIndex = 0; featureIndex < this.nFeaturesIn_!; featureIndex += 1) {
          value += X[i][featureIndex] * this.coef_![targetIndex][featureIndex];
        }
        row[targetIndex] = value;
      }
      predictions[i] = row;
    }
    return this.targetIsVector ? predictions.map((row) => row[0]) : predictions;
  }

  transform(X: Matrix): Matrix;
  transform(X: Matrix, Y: Matrix | Vector): [Matrix, Matrix];
  transform(X: Matrix, Y?: Matrix | Vector): TransformResult {
    this.assertFitted();
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }

    const xNormalized = normalizeWithMeanAndScale(X, this.xMean_!, this.xStd_!);
    const xScores = multiplyMatrices(xNormalized, this.xRotations_!);
    if (Y === undefined) {
      return xScores;
    }

    const target = toTargetMatrix(Y);
    if (target.Y.length !== X.length) {
      throw new Error(`Y must have the same number of rows as X. Expected ${X.length}, got ${target.Y.length}.`);
    }
    if (target.Y[0].length !== this.nTargetsIn_) {
      throw new Error(`Target size mismatch. Expected ${this.nTargetsIn_}, got ${target.Y[0].length}.`);
    }
    const yNormalized = normalizeWithMeanAndScale(target.Y, this.yMean_!, this.yStd_!);
    const yScores = multiplyMatrices(yNormalized, this.yRotations_!);
    return [xScores, yScores];
  }

  fitTransform(X: Matrix, Y: Matrix | Vector): Matrix {
    return this.fit(X, Y).transform(X);
  }

  inverseTransform(X: Matrix): Matrix {
    this.assertFitted();
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.xLoadings_![0].length) {
      throw new Error(
        `Component size mismatch. Expected ${this.xLoadings_![0].length}, got ${X[0].length}.`,
      );
    }
    const xApproxScaled = multiplyMatrices(X, transpose(this.xLoadings_!));
    return denormalizeWithMeanAndScale(xApproxScaled, this.xMean_!, this.xStd_!);
  }

  protected assertFitted(): void {
    if (
      !this.fitted ||
      !this.xWeights_ ||
      !this.yWeights_ ||
      !this.xLoadings_ ||
      !this.yLoadings_ ||
      !this.xRotations_ ||
      !this.yRotations_ ||
      !this.coef_ ||
      !this.intercept_ ||
      !this.xMean_ ||
      !this.yMean_ ||
      !this.xStd_ ||
      !this.yStd_ ||
      this.nFeaturesIn_ === null ||
      this.nTargetsIn_ === null
    ) {
      throw new Error("PLSRegression has not been fitted.");
    }
  }
}
