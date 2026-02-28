import type { Matrix, Vector } from "../types";
import { inverseMatrix, multiplyMatrices, transpose } from "../utils/linalg";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";
import { PCA } from "./PCA";

export interface FactorAnalysisOptions {
  nComponents?: number;
  tolerance?: number;
  maxIter?: number;
  randomState?: number;
}

function centerMatrix(X: Matrix, mean: Vector): Matrix {
  const out: Matrix = new Array(X.length);
  for (let i = 0; i < X.length; i += 1) {
    const row = new Array<number>(X[0].length);
    for (let j = 0; j < X[0].length; j += 1) {
      row[j] = X[i][j] - mean[j];
    }
    out[i] = row;
  }
  return out;
}

function featureVariances(XCentered: Matrix): Vector {
  const nSamples = XCentered.length;
  const nFeatures = XCentered[0].length;
  const out = new Array<number>(nFeatures).fill(0);
  const denominator = Math.max(1, nSamples - 1);
  for (let j = 0; j < nFeatures; j += 1) {
    let sum = 0;
    for (let i = 0; i < nSamples; i += 1) {
      const value = XCentered[i][j];
      sum += value * value;
    }
    out[j] = sum / denominator;
  }
  return out;
}

export class FactorAnalysis {
  components_: Matrix | null = null;
  mean_: Vector | null = null;
  noiseVariance_: Vector | null = null;
  nFeaturesIn_: number | null = null;
  nIter_: number | null = null;

  private nComponents?: number;
  private tolerance: number;
  private maxIter: number;
  private randomState?: number;
  private posteriorProjection_: Matrix | null = null;
  private fitted = false;

  constructor(options: FactorAnalysisOptions = {}) {
    this.nComponents = options.nComponents;
    this.tolerance = options.tolerance ?? 1e-6;
    this.maxIter = options.maxIter ?? 1000;
    this.randomState = options.randomState;

    if (this.nComponents !== undefined && (!Number.isInteger(this.nComponents) || this.nComponents < 1)) {
      throw new Error(`nComponents must be an integer >= 1 when provided. Got ${this.nComponents}.`);
    }
    if (!Number.isFinite(this.tolerance) || this.tolerance <= 0) {
      throw new Error(`tolerance must be finite and > 0. Got ${this.tolerance}.`);
    }
    if (!Number.isInteger(this.maxIter) || this.maxIter < 1) {
      throw new Error(`maxIter must be an integer >= 1. Got ${this.maxIter}.`);
    }
    if (this.randomState !== undefined && !Number.isFinite(this.randomState)) {
      throw new Error(`randomState must be finite when provided. Got ${this.randomState}.`);
    }
  }

  fit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    const nSamples = X.length;
    const nFeatures = X[0].length;
    const nComponents = Math.min(this.nComponents ?? nFeatures, nFeatures, nSamples);
    const pca = new PCA({
      nComponents,
      tolerance: this.tolerance,
      maxIter: this.maxIter,
    }).fit(X);

    const mean = pca.mean_!.slice();
    const centered = centerMatrix(X, mean);
    const featureVar = featureVariances(centered);
    const eigenvalues = pca.explainedVariance_!.slice();
    const eigenvectors = pca.components_!.map((row) => row.slice());

    let explainedSum = 0;
    for (let i = 0; i < eigenvalues.length; i += 1) {
      explainedSum += eigenvalues[i];
    }
    let totalVar = 0;
    for (let i = 0; i < featureVar.length; i += 1) {
      totalVar += featureVar[i];
    }
    const isotropicNoise = Math.max(1e-8, (totalVar - explainedSum) / Math.max(1, nFeatures));

    const components: Matrix = new Array(nComponents);
    for (let c = 0; c < nComponents; c += 1) {
      const loadingScale = Math.sqrt(Math.max(eigenvalues[c] - isotropicNoise, 1e-8));
      const row = new Array<number>(nFeatures);
      for (let f = 0; f < nFeatures; f += 1) {
        row[f] = eigenvectors[c][f] * loadingScale;
      }
      components[c] = row;
    }

    const noiseVariance = new Array<number>(nFeatures).fill(0);
    for (let f = 0; f < nFeatures; f += 1) {
      let explained = 0;
      for (let c = 0; c < nComponents; c += 1) {
        explained += components[c][f] * components[c][f];
      }
      noiseVariance[f] = Math.max(1e-8, featureVar[f] - explained);
    }

    const psiInv = noiseVariance.map((value) => 1 / Math.max(value, 1e-8));
    const wPsiInv = components.map((row) =>
      row.map((value, featureIndex) => value * psiInv[featureIndex]),
    );
    const factorCov = multiplyMatrices(wPsiInv, transpose(components));
    for (let i = 0; i < factorCov.length; i += 1) {
      factorCov[i][i] += 1;
    }
    const factorCovInv = inverseMatrix(factorCov);

    const psiInvWt: Matrix = Array.from({ length: nFeatures }, () => new Array<number>(nComponents).fill(0));
    for (let f = 0; f < nFeatures; f += 1) {
      for (let c = 0; c < nComponents; c += 1) {
        psiInvWt[f][c] = psiInv[f] * components[c][f];
      }
    }
    const posteriorProjection = multiplyMatrices(psiInvWt, factorCovInv);

    this.components_ = components;
    this.mean_ = mean;
    this.noiseVariance_ = noiseVariance;
    this.posteriorProjection_ = posteriorProjection;
    this.nFeaturesIn_ = nFeatures;
    this.nIter_ = 1;
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

    const centered = centerMatrix(X, this.mean_!);
    return multiplyMatrices(centered, this.posteriorProjection_!);
  }

  fitTransform(X: Matrix): Matrix {
    return this.fit(X).transform(X);
  }

  inverseTransform(X: Matrix): Matrix {
    this.assertFitted();
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.components_!.length) {
      throw new Error(`Component size mismatch. Expected ${this.components_!.length}, got ${X[0].length}.`);
    }

    const reconstructed = multiplyMatrices(X, this.components_!);
    for (let i = 0; i < reconstructed.length; i += 1) {
      for (let j = 0; j < reconstructed[i].length; j += 1) {
        reconstructed[i][j] += this.mean_![j];
      }
    }
    return reconstructed;
  }

  private assertFitted(): void {
    if (
      !this.fitted ||
      !this.components_ ||
      !this.mean_ ||
      !this.noiseVariance_ ||
      !this.posteriorProjection_ ||
      this.nFeaturesIn_ === null
    ) {
      throw new Error("FactorAnalysis has not been fitted.");
    }
  }
}
