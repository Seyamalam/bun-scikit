import type { ClassificationModel, Matrix, Vector } from "../types";
import { accuracyScore } from "../metrics/classification";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  validateClassificationInputs,
} from "../utils/validation";
import { argmax, buildLabelIndex, normalizeProbabilitiesInPlace, uniqueSortedLabels } from "../utils/classification";

export interface BernoulliNBOptions {
  alpha?: number;
  binarize?: number | null;
  fitPrior?: boolean;
  classPrior?: Vector | null;
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

export class BernoulliNB implements ClassificationModel {
  classes_: Vector = [];
  classCount_: Vector | null = null;
  classLogPrior_: Vector | null = null;
  featureCount_: Matrix | null = null;
  featureLogProb_: Matrix | null = null;
  nFeaturesIn_: number | null = null;

  private alpha: number;
  private binarize: number | null;
  private fitPrior: boolean;
  private classPrior: Vector | null;
  private fitted = false;

  constructor(options: BernoulliNBOptions = {}) {
    this.alpha = options.alpha ?? 1;
    this.binarize = options.binarize ?? 0;
    this.fitPrior = options.fitPrior ?? true;
    this.classPrior = options.classPrior ?? null;

    if (!Number.isFinite(this.alpha) || this.alpha < 0) {
      throw new Error(`alpha must be finite and >= 0. Got ${this.alpha}.`);
    }
    if (this.binarize !== null && !Number.isFinite(this.binarize)) {
      throw new Error(`binarize must be finite or null. Got ${this.binarize}.`);
    }
  }

  fit(X: Matrix, y: Vector): this {
    validateClassificationInputs(X, y);

    const nSamples = X.length;
    const nFeatures = X[0].length;
    const classes = uniqueSortedLabels(y);
    if (classes.length < 2) {
      throw new Error("BernoulliNB requires at least two classes.");
    }
    const classToIndex = buildLabelIndex(classes);

    const classCount = new Array<number>(classes.length).fill(0);
    const featureCount: Matrix = Array.from({ length: classes.length }, () => new Array<number>(nFeatures).fill(0));

    for (let i = 0; i < nSamples; i += 1) {
      const classIndex = classToIndex.get(y[i]);
      if (classIndex === undefined) {
        throw new Error(`Unknown label '${y[i]}' in fit targets.`);
      }
      classCount[classIndex] += 1;
      for (let j = 0; j < nFeatures; j += 1) {
        const raw = X[i][j];
        const value = this.binarize === null ? (raw > 0 ? 1 : 0) : (raw > this.binarize ? 1 : 0);
        featureCount[classIndex][j] += value;
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
      const denominator = classCount[c] + 2 * this.alpha;
      for (let j = 0; j < nFeatures; j += 1) {
        const probability = (featureCount[c][j] + this.alpha) / Math.max(denominator, 1e-12);
        featureLogProb[c][j] = Math.log(Math.min(1 - 1e-12, Math.max(1e-12, probability)));
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
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }

    const out: Matrix = new Array(X.length);
    for (let i = 0; i < X.length; i += 1) {
      const row = new Array<number>(this.classes_.length).fill(0);
      for (let c = 0; c < this.classes_.length; c += 1) {
        let score = this.classLogPrior_![c];
        for (let j = 0; j < this.nFeaturesIn_!; j += 1) {
          const raw = X[i][j];
          const xij = this.binarize === null ? (raw > 0 ? 1 : 0) : (raw > this.binarize ? 1 : 0);
          const lp = this.featureLogProb_![c][j];
          const negLp = Math.log(Math.max(1e-12, 1 - Math.exp(lp)));
          score += xij * lp + (1 - xij) * negLp;
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

  private assertFitted(): void {
    if (
      !this.fitted ||
      this.classCount_ === null ||
      this.classLogPrior_ === null ||
      this.featureCount_ === null ||
      this.featureLogProb_ === null ||
      this.nFeaturesIn_ === null
    ) {
      throw new Error("BernoulliNB has not been fitted.");
    }
  }
}

