import type { ClassificationModel, Matrix, Vector } from "../types";
import { accuracyScore } from "../metrics/classification";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  validateClassificationInputs,
} from "../utils/validation";
import { argmax, buildLabelIndex, normalizeProbabilitiesInPlace, uniqueSortedLabels } from "../utils/classification";

export interface CategoricalNBOptions {
  alpha?: number;
  fitPrior?: boolean;
  classPrior?: Vector | null;
  minCategories?: number[] | null;
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

export class CategoricalNB implements ClassificationModel {
  classes_: Vector = [];
  classCount_: Vector | null = null;
  classLogPrior_: Vector | null = null;
  categoryCount_: number[][][] | null = null;
  featureLogProb_: number[][][] | null = null;
  nCategories_: number[] | null = null;
  nFeaturesIn_: number | null = null;

  private alpha: number;
  private fitPrior: boolean;
  private classPrior: Vector | null;
  private minCategories: number[] | null;
  private fitted = false;

  constructor(options: CategoricalNBOptions = {}) {
    this.alpha = options.alpha ?? 1;
    this.fitPrior = options.fitPrior ?? true;
    this.classPrior = options.classPrior ?? null;
    this.minCategories = options.minCategories ?? null;
    if (!Number.isFinite(this.alpha) || this.alpha < 0) {
      throw new Error(`alpha must be finite and >= 0. Got ${this.alpha}.`);
    }
  }

  fit(X: Matrix, y: Vector): this {
    validateClassificationInputs(X, y);
    this.assertCategoricalIntegerMatrix(X);

    const nSamples = X.length;
    const nFeatures = X[0].length;
    const classes = uniqueSortedLabels(y);
    if (classes.length < 2) {
      throw new Error("CategoricalNB requires at least two classes.");
    }
    const classToIndex = buildLabelIndex(classes);

    if (this.minCategories && this.minCategories.length !== nFeatures) {
      throw new Error(`minCategories length must match nFeatures (${nFeatures}).`);
    }

    const nCategories = new Array<number>(nFeatures).fill(0);
    for (let j = 0; j < nFeatures; j += 1) {
      let maxValue = 0;
      for (let i = 0; i < nSamples; i += 1) {
        if (X[i][j] > maxValue) {
          maxValue = X[i][j];
        }
      }
      const inferred = maxValue + 1;
      const minCat = this.minCategories ? this.minCategories[j] : undefined;
      nCategories[j] = Math.max(inferred, minCat ?? inferred);
    }

    const classCount = new Array<number>(classes.length).fill(0);
    const categoryCount: number[][][] = Array.from({ length: classes.length }, () =>
      nCategories.map((count) => new Array<number>(count).fill(0)),
    );

    for (let i = 0; i < nSamples; i += 1) {
      const classIndex = classToIndex.get(y[i]);
      if (classIndex === undefined) {
        throw new Error(`Unknown label '${y[i]}' in fit targets.`);
      }
      classCount[classIndex] += 1;
      for (let j = 0; j < nFeatures; j += 1) {
        const category = X[i][j];
        categoryCount[classIndex][j][category] += 1;
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

    const featureLogProb: number[][][] = Array.from({ length: classes.length }, () =>
      nCategories.map((count) => new Array<number>(count).fill(0)),
    );
    for (let c = 0; c < classes.length; c += 1) {
      for (let j = 0; j < nFeatures; j += 1) {
        const denominator = classCount[c] + this.alpha * nCategories[j];
        for (let k = 0; k < nCategories[j]; k += 1) {
          featureLogProb[c][j][k] = Math.log((categoryCount[c][j][k] + this.alpha) / Math.max(denominator, 1e-12));
        }
      }
    }

    this.classes_ = classes;
    this.classCount_ = classCount;
    this.classLogPrior_ = classLogPrior;
    this.categoryCount_ = categoryCount;
    this.featureLogProb_ = featureLogProb;
    this.nCategories_ = nCategories;
    this.nFeaturesIn_ = nFeatures;
    this.fitted = true;
    return this;
  }

  predictLogProba(X: Matrix): Matrix {
    this.assertFitted();
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    this.assertCategoricalIntegerMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }

    const out: Matrix = new Array(X.length);
    for (let i = 0; i < X.length; i += 1) {
      const row = new Array<number>(this.classes_.length).fill(0);
      for (let c = 0; c < this.classes_.length; c += 1) {
        let score = this.classLogPrior_![c];
        for (let j = 0; j < this.nFeaturesIn_!; j += 1) {
          const category = X[i][j];
          const categoryCount = this.nCategories_![j];
          if (category >= categoryCount) {
            score += Math.log(this.alpha / Math.max(this.classCount_![c] + this.alpha * categoryCount, 1e-12));
          } else {
            score += this.featureLogProb_![c][j][category];
          }
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

  private assertCategoricalIntegerMatrix(X: Matrix): void {
    for (let i = 0; i < X.length; i += 1) {
      for (let j = 0; j < X[i].length; j += 1) {
        const value = X[i][j];
        if (!Number.isInteger(value) || value < 0) {
          throw new Error(
            `CategoricalNB expects non-negative integer categories. Found ${value} at [${i}, ${j}].`,
          );
        }
      }
    }
  }

  private assertFitted(): void {
    if (
      !this.fitted ||
      this.classCount_ === null ||
      this.classLogPrior_ === null ||
      this.categoryCount_ === null ||
      this.featureLogProb_ === null ||
      this.nCategories_ === null ||
      this.nFeaturesIn_ === null
    ) {
      throw new Error("CategoricalNB has not been fitted.");
    }
  }
}

