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
  logDeterminant,
  quadraticForm,
  regularizedInverse,
} from "./shared";

export interface QuadraticDiscriminantAnalysisOptions {
  priors?: number[];
  regParam?: number;
}

export class QuadraticDiscriminantAnalysis implements ClassificationModel {
  classes_: Vector = [0, 1];
  means_: Matrix | null = null;
  priors_: Vector | null = null;
  covariances_: Matrix[] | null = null;

  private precisions_: Matrix[] | null = null;
  private logDets_: Vector | null = null;
  private nFeaturesIn_: number | null = null;
  private fitted = false;

  private priors?: number[];
  private regParam: number;

  constructor(options: QuadraticDiscriminantAnalysisOptions = {}) {
    this.priors = options.priors;
    this.regParam = options.regParam ?? 0;
    if (!Number.isFinite(this.regParam) || this.regParam < 0 || this.regParam > 1) {
      throw new Error(`regParam must be finite and in [0, 1]. Got ${this.regParam}.`);
    }
  }

  fit(X: Matrix, y: Vector, sampleWeight?: Vector): this {
    validateClassificationInputs(X, y);
    this.classes_ = uniqueSortedLabels(y);
    if (this.classes_.length < 2) {
      throw new Error("QuadraticDiscriminantAnalysis requires at least two classes.");
    }

    const byClass = classIndices(y);
    const nSamples = X.length;
    const nFeatures = X[0].length;

    const means: Matrix = new Array(this.classes_.length);
    const priors = new Array<number>(this.classes_.length).fill(0);
    const covariances: Matrix[] = new Array(this.classes_.length);
    const precisions: Matrix[] = new Array(this.classes_.length);
    const logDets = new Array<number>(this.classes_.length).fill(0);

    for (let c = 0; c < this.classes_.length; c += 1) {
      const label = this.classes_[c];
      const indices = byClass.get(label) ?? [];
      const Xc = indices.map((idx) => X[idx]);
      const mean = featureMeans(Xc);
      let cov = covariance(Xc, mean);
      if (this.regParam > 0) {
        for (let i = 0; i < cov.length; i += 1) {
          cov[i][i] = (1 - this.regParam) * cov[i][i] + this.regParam;
        }
      }
      const precision = regularizedInverse(cov);
      const logDet = logDeterminant(cov);

      means[c] = mean;
      priors[c] = indices.length / nSamples;
      covariances[c] = cov;
      precisions[c] = precision;
      logDets[c] = logDet;
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
      for (let i = 0; i < this.priors.length; i += 1) {
        priors[i] = this.priors[i] / sum;
      }
    }

    this.means_ = means;
    this.priors_ = priors;
    this.covariances_ = covariances;
    this.precisions_ = precisions;
    this.logDets_ = logDets;
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
        const diff = new Array<number>(this.nFeaturesIn_!);
        for (let j = 0; j < this.nFeaturesIn_!; j += 1) {
          diff[j] = X[i][j] - this.means_![c][j];
        }
        const quad = quadraticForm(diff, this.precisions_![c]);
        scores[c] = -0.5 * (this.logDets_![c] + quad) + Math.log(this.priors_![c]);
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
      !this.covariances_ ||
      !this.precisions_ ||
      !this.logDets_ ||
      this.nFeaturesIn_ === null
    ) {
      throw new Error("QuadraticDiscriminantAnalysis has not been fitted.");
    }
  }
}
