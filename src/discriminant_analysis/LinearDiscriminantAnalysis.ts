import type { ClassificationModel, Matrix, Vector } from "../types";
import { accuracyScore } from "../metrics/classification";
import { argmax, normalizeProbabilitiesInPlace, uniqueSortedLabels } from "../utils/classification";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  validateClassificationInputs,
} from "../utils/validation";
import {
  classIndices,
  covariance,
  featureMeans,
  quadraticForm,
  regularizedInverse,
} from "./shared";

export interface LinearDiscriminantAnalysisOptions {
  priors?: number[];
}

export class LinearDiscriminantAnalysis implements ClassificationModel {
  classes_: Vector = [0, 1];
  means_: Matrix | null = null;
  priors_: Vector | null = null;
  covariance_: Matrix | null = null;

  private coef_: Matrix | null = null;
  private intercept_: Vector | null = null;
  private nFeaturesIn_: number | null = null;
  private fitted = false;

  private priors?: number[];

  constructor(options: LinearDiscriminantAnalysisOptions = {}) {
    this.priors = options.priors;
  }

  fit(X: Matrix, y: Vector, sampleWeight?: Vector): this {
    validateClassificationInputs(X, y);
    this.classes_ = uniqueSortedLabels(y);
    const byClass = classIndices(y);
    const nSamples = X.length;
    const nFeatures = X[0].length;

    if (this.classes_.length < 2) {
      throw new Error("LinearDiscriminantAnalysis requires at least two classes.");
    }

    const means: Matrix = new Array(this.classes_.length);
    const classPriors = new Array<number>(this.classes_.length).fill(0);
    const pooled: Matrix = Array.from({ length: nFeatures }, () =>
      new Array<number>(nFeatures).fill(0),
    );

    for (let c = 0; c < this.classes_.length; c += 1) {
      const label = this.classes_[c];
      const indices = byClass.get(label) ?? [];
      const Xc = indices.map((idx) => X[idx]);
      const mean = featureMeans(Xc);
      means[c] = mean;
      classPriors[c] = indices.length / nSamples;
      const cov = covariance(Xc, mean);
      for (let i = 0; i < nFeatures; i += 1) {
        for (let j = 0; j < nFeatures; j += 1) {
          pooled[i][j] += cov[i][j] * Math.max(1, Xc.length - 1);
        }
      }
    }

    const denom = Math.max(1, nSamples - this.classes_.length);
    for (let i = 0; i < nFeatures; i += 1) {
      for (let j = 0; j < nFeatures; j += 1) {
        pooled[i][j] /= denom;
      }
    }

    if (this.priors !== undefined) {
      if (this.priors.length !== this.classes_.length) {
        throw new Error(`priors length must match number of classes ${this.classes_.length}.`);
      }
      let sum = 0;
      for (let i = 0; i < this.priors.length; i += 1) {
        const p = this.priors[i];
        if (!Number.isFinite(p) || p <= 0) {
          throw new Error(`priors must contain finite positive values. Got ${p}.`);
        }
        sum += p;
      }
      classPriors.fill(0);
      for (let i = 0; i < this.priors.length; i += 1) {
        classPriors[i] = this.priors[i] / sum;
      }
    }

    const precision = regularizedInverse(pooled);
    const coef: Matrix = new Array(this.classes_.length);
    const intercept = new Array<number>(this.classes_.length).fill(0);

    for (let c = 0; c < this.classes_.length; c += 1) {
      const mean = means[c];
      const w = new Array<number>(nFeatures).fill(0);
      for (let i = 0; i < nFeatures; i += 1) {
        for (let j = 0; j < nFeatures; j += 1) {
          w[i] += precision[i][j] * mean[j];
        }
      }
      coef[c] = w;
      intercept[c] = -0.5 * quadraticForm(mean, precision) + Math.log(classPriors[c]);
    }

    this.means_ = means;
    this.priors_ = classPriors;
    this.covariance_ = pooled;
    this.coef_ = coef;
    this.intercept_ = intercept;
    this.nFeaturesIn_ = nFeatures;
    this.fitted = true;
    return this;
  }

  predictProba(X: Matrix): Matrix {
    this.assertFitted();
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }

    const out: Matrix = new Array(X.length);
    for (let i = 0; i < X.length; i += 1) {
      const scores = new Array<number>(this.classes_.length).fill(0);
      for (let c = 0; c < this.classes_.length; c += 1) {
        let score = this.intercept_![c];
        for (let j = 0; j < this.nFeaturesIn_!; j += 1) {
          score += this.coef_![c][j] * X[i][j];
        }
        scores[c] = score;
      }
      let maxScore = scores[0];
      for (let c = 1; c < scores.length; c += 1) {
        if (scores[c] > maxScore) {
          maxScore = scores[c];
        }
      }
      for (let c = 0; c < scores.length; c += 1) {
        scores[c] = Math.exp(scores[c] - maxScore);
      }
      normalizeProbabilitiesInPlace(scores);
      out[i] = scores;
    }
    return out;
  }

  predict(X: Matrix): Vector {
    return this.predictProba(X).map((row) => this.classes_[argmax(row)]);
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return accuracyScore(y, this.predict(X));
  }

  private assertFitted(): void {
    if (
      !this.fitted ||
      !this.means_ ||
      !this.priors_ ||
      !this.covariance_ ||
      !this.coef_ ||
      !this.intercept_ ||
      this.nFeaturesIn_ === null
    ) {
      throw new Error("LinearDiscriminantAnalysis has not been fitted.");
    }
  }
}
