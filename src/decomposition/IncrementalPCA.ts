import type { Matrix, Vector } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";
import { PCA, type PCAOptions } from "./PCA";

export interface IncrementalPCAOptions extends PCAOptions {
  batchSize?: number;
}

export class IncrementalPCA {
  components_: Matrix | null = null;
  explainedVariance_: Vector | null = null;
  explainedVarianceRatio_: Vector | null = null;
  singularValues_: Vector | null = null;
  mean_: Vector | null = null;
  nFeaturesIn_: number | null = null;
  nSamplesSeen_: number = 0;

  private nComponents?: number;
  private whiten: boolean;
  private tolerance: number;
  private maxIter: number;
  private batchSize: number | null;
  private pcaModel: PCA | null = null;
  private dataBuffer: Matrix = [];
  private fitted = false;

  constructor(options: IncrementalPCAOptions = {}) {
    this.nComponents = options.nComponents;
    this.whiten = options.whiten ?? false;
    this.tolerance = options.tolerance ?? 1e-12;
    this.maxIter = options.maxIter ?? 10_000;
    this.batchSize = options.batchSize ?? null;

    if (this.batchSize !== null && (!Number.isInteger(this.batchSize) || this.batchSize < 1)) {
      throw new Error(`batchSize must be an integer >= 1 when provided. Got ${this.batchSize}.`);
    }
  }

  fit(X: Matrix): this {
    this.dataBuffer = [];
    this.nSamplesSeen_ = 0;
    return this.partialFit(X);
  }

  partialFit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    if (this.nFeaturesIn_ !== null && X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }
    this.nFeaturesIn_ = X[0].length;

    for (let i = 0; i < X.length; i += 1) {
      this.dataBuffer.push(X[i].slice());
    }
    this.nSamplesSeen_ += X.length;

    const model = new PCA({
      nComponents: this.nComponents,
      whiten: this.whiten,
      tolerance: this.tolerance,
      maxIter: this.maxIter,
    }).fit(this.dataBuffer);

    this.pcaModel = model;
    this.components_ = model.components_ ? model.components_.map((row) => row.slice()) : null;
    this.explainedVariance_ = model.explainedVariance_ ? model.explainedVariance_.slice() : null;
    this.explainedVarianceRatio_ = model.explainedVarianceRatio_ ? model.explainedVarianceRatio_.slice() : null;
    this.singularValues_ = model.singularValues_ ? model.singularValues_.slice() : null;
    this.mean_ = model.mean_ ? model.mean_.slice() : null;
    this.fitted = true;
    return this;
  }

  transform(X: Matrix): Matrix {
    this.assertFitted();
    return this.pcaModel!.transform(X);
  }

  fitTransform(X: Matrix): Matrix {
    return this.fit(X).transform(X);
  }

  inverseTransform(X: Matrix): Matrix {
    this.assertFitted();
    return this.pcaModel!.inverseTransform(X);
  }

  private assertFitted(): void {
    if (!this.fitted || !this.pcaModel || !this.components_ || this.nFeaturesIn_ === null) {
      throw new Error("IncrementalPCA has not been fitted.");
    }
  }
}
