import type { ClassificationModel, Matrix, Vector } from "../types";
import { accuracyScore } from "../metrics/classification";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  validateClassificationInputs,
} from "../utils/validation";
import { argmax, buildLabelIndex, normalizeProbabilitiesInPlace, uniqueSortedLabels } from "../utils/classification";

export interface ComplementNBOptions {
  alpha?: number;
  fitPrior?: boolean;
  classPrior?: Vector | null;
  norm?: boolean;
}

function logSumExp(values: Vector): number {
  let maxValue = values[0];
  for (let i = 1; i < values.length; i += 1) {
    if (values[i] > maxValue) {
      maxValue = values[i];
    }
  }
  let sum = 0;
  for (let i = 0; i < values.length; i += 1) {
    sum += Math.exp(values[i] - maxValue);
  }
  return maxValue + Math.log(sum);
}

export class ComplementNB implements ClassificationModel {
  classes_: Vector = [];
  classCount_: Vector | null = null;
  classLogPrior_: Vector | null = null;
  featureCount_: Matrix | null = null;
  featureLogProb_: Matrix | null = null;
  nFeaturesIn_: number | null = null;

  private alpha: number;
  private fitPrior: boolean;
  private classPrior: Vector | null;
  private norm: boolean;
  private fitted = false;

  constructor(options: ComplementNBOptions = {}) {
    this.alpha = options.alpha ?? 1;
    this.fitPrior = options.fitPrior ?? true;
    this.classPrior = options.classPrior ?? null;
    this.norm = options.norm ?? false;
    if (!Number.isFinite(this.alpha) || this.alpha < 0) {
      throw new Error(`alpha must be finite and >= 0. Got ${this.alpha}.`);
    }
  }

  fit(X: Matrix, y: Vector): this {
    validateClassificationInputs(X, y);
    this.assertNonNegativeMatrix(X);

    const nSamples = X.length;
    const nFeatures = X[0].length;
    const classes = uniqueSortedLabels(y);
    if (classes.length < 2) {
      throw new Error("ComplementNB requires at least two classes.");
    }
    const classToIndex = buildLabelIndex(classes);

    const classCount = new Array<number>(classes.length).fill(0);
    const featureCount: Matrix = Array.from({ length: classes.length }, () => new Array<number>(nFeatures).fill(0));
    const totalFeatureCount = new Array<number>(nFeatures).fill(0);

    for (let i = 0; i < nSamples; i += 1) {
      const classIndex = classToIndex.get(y[i]);
      if (classIndex === undefined) {
        throw new Error(`Unknown label '${y[i]}' in fit targets.`);
      }
      classCount[classIndex] += 1;
      for (let j = 0; j < nFeatures; j += 1) {
        const value = X[i][j];
        featureCount[classIndex][j] += value;
        totalFeatureCount[j] += value;
      }
    }

    const classLogPrior = new Array<number>(classes.length).fill(0);
    if (this.classPrior) {
      if (this.classPrior.length !== classes.length) {
        throw new Error(`classPrior length must match class count ${classes.length}.`);
      }
      for (let i = 0; i < classes.length; i += 1) {
        if (!Number.isFinite(this.classPrior[i]) || this.classPrior[i] <= 0) {
          throw new Error("classPrior values must be finite and > 0.");
        }
        classLogPrior[i] = Math.log(this.classPrior[i]);
      }
    } else if (this.fitPrior) {
      for (let i = 0; i < classes.length; i += 1) {
        classLogPrior[i] = Math.log(classCount[i] / nSamples);
      }
    } else {
      const uniform = -Math.log(classes.length);
      for (let i = 0; i < classes.length; i += 1) {
        classLogPrior[i] = uniform;
      }
    }

    const featureLogProb: Matrix = Array.from({ length: classes.length }, () => new Array<number>(nFeatures).fill(0));
    for (let c = 0; c < classes.length; c += 1) {
      let complementTotal = 0;
      const complement = new Array<number>(nFeatures).fill(0);
      for (let j = 0; j < nFeatures; j += 1) {
        complement[j] = totalFeatureCount[j] - featureCount[c][j];
        complementTotal += complement[j];
      }
      const denominator = complementTotal + this.alpha * nFeatures;
      let absSum = 0;
      for (let j = 0; j < nFeatures; j += 1) {
        const value = Math.log((complement[j] + this.alpha) / Math.max(denominator, 1e-12));
        featureLogProb[c][j] = value;
        absSum += Math.abs(value);
      }
      if (this.norm && absSum > 0) {
        for (let j = 0; j < nFeatures; j += 1) {
          featureLogProb[c][j] /= absSum;
        }
      }
    }

    this.classes_ = classes;
    this.classCount_ = classCount;
    this.classLogPrior_ = classLogPrior;
    this.featureCount_ = featureCount;
    this.featureLogProb_ = featureLogProb;
    this.nFeaturesIn_ = nFeatures;
    this.fitted = true;
    return this;
  }

  predictLogProba(X: Matrix): Matrix {
    this.assertFitted();
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    this.assertNonNegativeMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }

    const out: Matrix = new Array(X.length);
    for (let i = 0; i < X.length; i += 1) {
      const row = new Array<number>(this.classes_.length).fill(0);
      for (let c = 0; c < this.classes_.length; c += 1) {
        let score = this.classLogPrior_![c];
        for (let j = 0; j < this.nFeaturesIn_!; j += 1) {
          // Higher complement probability means less likely for target class.
          score -= X[i][j] * this.featureLogProb_![c][j];
        }
        row[c] = score;
      }
      const logNorm = logSumExp(row);
      for (let c = 0; c < row.length; c += 1) {
        row[c] -= logNorm;
      }
      out[i] = row;
    }
    return out;
  }

  predictProba(X: Matrix): Matrix {
    const logProba = this.predictLogProba(X);
    for (let i = 0; i < logProba.length; i += 1) {
      for (let j = 0; j < logProba[i].length; j += 1) {
        logProba[i][j] = Math.exp(logProba[i][j]);
      }
      normalizeProbabilitiesInPlace(logProba[i]);
    }
    return logProba;
  }

  predict(X: Matrix): Vector {
    const probabilities = this.predictProba(X);
    return probabilities.map((row) => this.classes_[argmax(row)]);
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return accuracyScore(y, this.predict(X));
  }

  private assertNonNegativeMatrix(X: Matrix): void {
    for (let i = 0; i < X.length; i += 1) {
      for (let j = 0; j < X[i].length; j += 1) {
        if (X[i][j] < 0) {
          throw new Error(`ComplementNB expects non-negative features. Found ${X[i][j]} at [${i}, ${j}].`);
        }
      }
    }
  }

  private assertFitted(): void {
    if (
      !this.fitted ||
      this.classCount_ === null ||
      this.classLogPrior_ === null ||
      this.featureCount_ === null ||
      this.featureLogProb_ === null ||
      this.nFeaturesIn_ === null
    ) {
      throw new Error("ComplementNB has not been fitted.");
    }
  }
}

