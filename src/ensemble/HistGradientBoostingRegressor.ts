import type { Matrix, Vector } from "../types";
import { r2Score } from "../metrics/regression";
import { assertFiniteVector, validateRegressionInputs } from "../utils/validation";

export interface HistGradientBoostingRegressorOptions {
  maxIter?: number;
  learningRate?: number;
  maxBins?: number;
  minSamplesLeaf?: number;
  l2Regularization?: number;
  randomState?: number;
}

type HistStump =
  | {
      kind: "constant";
      value: number;
    }
  | {
      kind: "split";
      featureIndex: number;
      thresholdBin: number;
      leftValue: number;
      rightValue: number;
    };

function mean(y: Vector): number {
  let total = 0;
  for (let i = 0; i < y.length; i += 1) {
    total += y[i];
  }
  return total / y.length;
}

function digitize(value: number, thresholds: number[]): number {
  let lo = 0;
  let hi = thresholds.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (value <= thresholds[mid]) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  return lo;
}

function buildFeatureThresholds(X: Matrix, maxBins: number): number[][] {
  const nFeatures = X[0].length;
  const out: number[][] = new Array<number[]>(nFeatures);
  for (let feature = 0; feature < nFeatures; feature += 1) {
    const values = new Array<number>(X.length);
    for (let i = 0; i < X.length; i += 1) {
      values[i] = X[i][feature];
    }
    values.sort((a, b) => a - b);

    const uniqueCount =
      values.length === 0 ? 0 : 1 + values.reduce((acc, value, idx) => acc + (idx > 0 && value !== values[idx - 1] ? 1 : 0), 0);
    const bins = Math.max(1, Math.min(maxBins, uniqueCount));
    const thresholds: number[] = [];
    for (let b = 1; b < bins; b += 1) {
      const q = (b * values.length) / bins;
      const rightIdx = Math.min(values.length - 1, Math.max(1, Math.floor(q)));
      const leftIdx = rightIdx - 1;
      const threshold = (values[leftIdx] + values[rightIdx]) / 2;
      if (thresholds.length === 0 || threshold > thresholds[thresholds.length - 1]) {
        thresholds.push(threshold);
      }
    }
    out[feature] = thresholds;
  }
  return out;
}

export class HistGradientBoostingRegressor {
  estimators_: HistStump[] = [];
  baseline_: number | null = null;
  binThresholds_: number[][] = [];

  private readonly maxIter: number;
  private readonly learningRate: number;
  private readonly maxBins: number;
  private readonly minSamplesLeaf: number;
  private readonly l2Regularization: number;
  private readonly randomState?: number;
  private isFitted = false;

  constructor(options: HistGradientBoostingRegressorOptions = {}) {
    this.maxIter = options.maxIter ?? 100;
    this.learningRate = options.learningRate ?? 0.1;
    this.maxBins = options.maxBins ?? 255;
    this.minSamplesLeaf = options.minSamplesLeaf ?? 20;
    this.l2Regularization = options.l2Regularization ?? 0;
    this.randomState = options.randomState;

    if (!Number.isInteger(this.maxIter) || this.maxIter < 1) {
      throw new Error(`maxIter must be an integer >= 1. Got ${this.maxIter}.`);
    }
    if (!Number.isFinite(this.learningRate) || this.learningRate <= 0) {
      throw new Error(`learningRate must be finite and > 0. Got ${this.learningRate}.`);
    }
    if (!Number.isInteger(this.maxBins) || this.maxBins < 2) {
      throw new Error(`maxBins must be an integer >= 2. Got ${this.maxBins}.`);
    }
    if (!Number.isInteger(this.minSamplesLeaf) || this.minSamplesLeaf < 1) {
      throw new Error(
        `minSamplesLeaf must be an integer >= 1. Got ${this.minSamplesLeaf}.`,
      );
    }
    if (!Number.isFinite(this.l2Regularization) || this.l2Regularization < 0) {
      throw new Error(
        `l2Regularization must be finite and >= 0. Got ${this.l2Regularization}.`,
      );
    }
  }

  fit(X: Matrix, y: Vector): this {
    validateRegressionInputs(X, y);
    const nSamples = X.length;
    this.estimators_ = [];
    this.binThresholds_ = buildFeatureThresholds(X, this.maxBins);

    const binned: number[][] = Array.from({ length: nSamples }, () =>
      new Array<number>(X[0].length).fill(0),
    );
    for (let i = 0; i < nSamples; i += 1) {
      for (let feature = 0; feature < X[i].length; feature += 1) {
        binned[i][feature] = digitize(X[i][feature], this.binThresholds_[feature]);
      }
    }

    this.baseline_ = mean(y);
    const prediction = new Array<number>(nSamples).fill(this.baseline_);

    for (let iter = 0; iter < this.maxIter; iter += 1) {
      const residuals = new Array<number>(nSamples);
      for (let i = 0; i < nSamples; i += 1) {
        residuals[i] = y[i] - prediction[i];
      }

      const stump = this.fitStump(residuals, binned);
      this.estimators_.push(stump);

      for (let i = 0; i < nSamples; i += 1) {
        prediction[i] += this.learningRate * this.predictStumpBinned(stump, binned[i]);
      }
    }

    this.isFitted = true;
    return this;
  }

  predict(X: Matrix): Vector {
    this.assertFitted();
    const out = new Array<number>(X.length).fill(this.baseline_!);
    for (let t = 0; t < this.estimators_.length; t += 1) {
      const stump = this.estimators_[t];
      for (let row = 0; row < X.length; row += 1) {
        out[row] += this.learningRate * this.predictStumpRow(stump, X[row]);
      }
    }
    return out;
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return r2Score(y, this.predict(X));
  }

  private fitStump(residuals: Vector, binned: number[][]): HistStump {
    const nSamples = residuals.length;
    const nFeatures = binned[0].length;

    let totalSum = 0;
    let totalSq = 0;
    for (let i = 0; i < nSamples; i += 1) {
      totalSum += residuals[i];
      totalSq += residuals[i] * residuals[i];
    }

    let bestScore =
      totalSq - (totalSum * totalSum) / (nSamples + this.l2Regularization);
    let best: HistStump = {
      kind: "constant",
      value: totalSum / (nSamples + this.l2Regularization),
    };

    for (let feature = 0; feature < nFeatures; feature += 1) {
      const binCount = this.binThresholds_[feature].length + 1;
      if (binCount < 2) {
        continue;
      }

      const counts = new Array<number>(binCount).fill(0);
      const sums = new Array<number>(binCount).fill(0);
      const sumSquares = new Array<number>(binCount).fill(0);
      for (let i = 0; i < nSamples; i += 1) {
        const bin = binned[i][feature];
        const residual = residuals[i];
        counts[bin] += 1;
        sums[bin] += residual;
        sumSquares[bin] += residual * residual;
      }

      let leftCount = 0;
      let leftSum = 0;
      let leftSq = 0;
      for (let splitBin = 0; splitBin < binCount - 1; splitBin += 1) {
        leftCount += counts[splitBin];
        leftSum += sums[splitBin];
        leftSq += sumSquares[splitBin];
        const rightCount = nSamples - leftCount;
        if (leftCount < this.minSamplesLeaf || rightCount < this.minSamplesLeaf) {
          continue;
        }

        const rightSum = totalSum - leftSum;
        const rightSq = totalSq - leftSq;
        const leftLoss = leftSq - (leftSum * leftSum) / (leftCount + this.l2Regularization);
        const rightLoss =
          rightSq - (rightSum * rightSum) / (rightCount + this.l2Regularization);
        const score = leftLoss + rightLoss;
        if (score + 1e-12 < bestScore) {
          bestScore = score;
          best = {
            kind: "split",
            featureIndex: feature,
            thresholdBin: splitBin,
            leftValue: leftSum / (leftCount + this.l2Regularization),
            rightValue: rightSum / (rightCount + this.l2Regularization),
          };
        }
      }
    }

    return best;
  }

  private predictStumpBinned(stump: HistStump, bins: number[]): number {
    if (stump.kind === "constant") {
      return stump.value;
    }
    return bins[stump.featureIndex] <= stump.thresholdBin ? stump.leftValue : stump.rightValue;
  }

  private predictStumpRow(stump: HistStump, row: number[]): number {
    if (stump.kind === "constant") {
      return stump.value;
    }
    const bin = digitize(row[stump.featureIndex], this.binThresholds_[stump.featureIndex]);
    return bin <= stump.thresholdBin ? stump.leftValue : stump.rightValue;
  }

  private assertFitted(): void {
    if (!this.isFitted || this.baseline_ === null) {
      throw new Error("HistGradientBoostingRegressor has not been fitted.");
    }
  }
}
