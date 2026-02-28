import type { Matrix, RegressionModel, Vector } from "../types";
import { r2Score } from "../metrics/regression";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  validateRegressionInputs,
} from "../utils/validation";
import type { MaxFeaturesOption } from "./DecisionTreeClassifier";

export interface DecisionTreeRegressorOptions {
  maxDepth?: number;
  minSamplesSplit?: number;
  minSamplesLeaf?: number;
  maxFeatures?: MaxFeaturesOption;
  splitter?: "best" | "random";
  randomState?: number;
}

interface TreeNode {
  prediction: number;
  featureIndex?: number;
  threshold?: number;
  left?: TreeNode;
  right?: TreeNode;
  isLeaf: boolean;
}

interface SplitEvaluation {
  threshold: number;
  impurity: number;
}

interface SplitPartition {
  leftIndices: number[];
  rightIndices: number[];
}

const MAX_THRESHOLD_BINS = 128;

function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function safeVariance(sum: number, sumSquares: number, count: number): number {
  if (count <= 0) {
    return 0;
  }
  const mean = sum / count;
  const variance = sumSquares / count - mean * mean;
  return variance < 0 ? 0 : variance;
}

export class DecisionTreeRegressor implements RegressionModel {
  featureImportances_: Vector | null = null;
  private readonly maxDepth: number;
  private readonly minSamplesSplit: number;
  private readonly minSamplesLeaf: number;
  private readonly maxFeatures: MaxFeaturesOption;
  private readonly splitter: "best" | "random";
  private readonly randomState?: number;
  private random: () => number = Math.random;
  private root: TreeNode | null = null;
  private flattenedXTrain: Float64Array | null = null;
  private yTrain: Float64Array | null = null;
  private featureCount = 0;
  private allFeatureIndices: number[] = [];
  private featureSelectionMarks: Uint8Array | null = null;
  private binTotals: Uint32Array = new Uint32Array(MAX_THRESHOLD_BINS);
  private binSums: Float64Array = new Float64Array(MAX_THRESHOLD_BINS);
  private binSumsSquares: Float64Array = new Float64Array(MAX_THRESHOLD_BINS);
  private featureImportanceRaw: Float64Array | null = null;

  constructor(options: DecisionTreeRegressorOptions = {}) {
    this.maxDepth = options.maxDepth ?? 12;
    this.minSamplesSplit = options.minSamplesSplit ?? 2;
    this.minSamplesLeaf = options.minSamplesLeaf ?? 1;
    this.maxFeatures = options.maxFeatures ?? null;
    this.splitter = options.splitter ?? "best";
    this.randomState = options.randomState;
    if (this.splitter !== "best" && this.splitter !== "random") {
      throw new Error(`splitter must be 'best' or 'random'. Got ${this.splitter}.`);
    }
  }

  fit(
    X: Matrix,
    y: Vector,
    sampleIndices?: ArrayLike<number>,
    skipValidation = false,
    flattenedXTrain?: Float64Array,
    yTrain?: Float64Array,
  ): this {
    if (!skipValidation) {
      validateRegressionInputs(X, y);
    }

    this.featureCount = X[0].length;
    this.featureImportanceRaw = new Float64Array(this.featureCount);
    this.featureImportances_ = null;
    this.flattenedXTrain = flattenedXTrain ?? this.flattenTrainingMatrix(X);
    this.yTrain = yTrain ?? this.toFloat64Vector(y);
    this.allFeatureIndices = new Array<number>(this.featureCount);
    for (let i = 0; i < this.featureCount; i += 1) {
      this.allFeatureIndices[i] = i;
    }
    this.featureSelectionMarks = new Uint8Array(this.featureCount);
    this.random = this.randomState === undefined ? Math.random : mulberry32(this.randomState);

    let rootIndices: number[];
    if (sampleIndices) {
      if (sampleIndices.length === 0) {
        throw new Error("sampleIndices must not be empty.");
      }
      rootIndices = new Array<number>(sampleIndices.length);
      for (let i = 0; i < sampleIndices.length; i += 1) {
        const index = sampleIndices[i];
        if (!Number.isInteger(index) || index < 0 || index >= X.length) {
          throw new Error(`sampleIndices contains invalid index: ${index}.`);
        }
        rootIndices[i] = index;
      }
    } else {
      rootIndices = new Array<number>(X.length);
      for (let i = 0; i < X.length; i += 1) {
        rootIndices[i] = i;
      }
    }

    this.root = this.buildTree(rootIndices, 0);
    this.finalizeFeatureImportances();
    return this;
  }

  predict(X: Matrix): Vector {
    if (!this.root || this.featureCount === 0) {
      throw new Error("DecisionTreeRegressor has not been fitted.");
    }

    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    if (X[0].length !== this.featureCount) {
      throw new Error(
        `Feature size mismatch. Expected ${this.featureCount}, got ${X[0].length}.`,
      );
    }

    return X.map((sample) => this.predictOne(sample, this.root!));
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return r2Score(y, this.predict(X));
  }

  private predictOne(sample: Vector, node: TreeNode): number {
    let current = node;
    while (
      !current.isLeaf &&
      current.featureIndex !== undefined &&
      current.threshold !== undefined
    ) {
      if (sample[current.featureIndex] <= current.threshold) {
        current = current.left!;
      } else {
        current = current.right!;
      }
    }
    return current.prediction;
  }

  private buildTree(indices: number[], depth: number): TreeNode {
    const y = this.yTrain!;
    const sampleCount = indices.length;

    let sum = 0;
    let sumSquares = 0;
    for (let i = 0; i < sampleCount; i += 1) {
      const value = y[indices[i]];
      sum += value;
      sumSquares += value * value;
    }
    const prediction = sum / sampleCount;
    const parentVariance = safeVariance(sum, sumSquares, sampleCount);

    const depthStop = depth >= this.maxDepth;
    const splitStop = sampleCount < this.minSamplesSplit;
    const pureEnough = parentVariance <= 1e-14;
    if (depthStop || splitStop || pureEnough) {
      return { isLeaf: true, prediction };
    }

    const candidateFeatures = this.selectFeatureIndices(this.featureCount);
    let bestFeature = -1;
    let bestSplit: SplitEvaluation | null = null;

    for (let i = 0; i < candidateFeatures.length; i += 1) {
      const featureIndex = candidateFeatures[i];
      const split = this.findBestThreshold(indices, featureIndex);
      if (!split) {
        continue;
      }
      if (!bestSplit || split.impurity < bestSplit.impurity) {
        bestFeature = featureIndex;
        bestSplit = split;
      }
    }

    if (!bestSplit || bestFeature === -1 || bestSplit.impurity >= parentVariance - 1e-14) {
      return { isLeaf: true, prediction };
    }

    const partition = this.partitionIndices(indices, bestFeature, bestSplit.threshold);
    if (!partition) {
      return { isLeaf: true, prediction };
    }
    const gain = parentVariance - bestSplit.impurity;
    if (gain > 0 && this.featureImportanceRaw) {
      this.featureImportanceRaw[bestFeature] += sampleCount * gain;
    }

    return {
      isLeaf: false,
      prediction,
      featureIndex: bestFeature,
      threshold: bestSplit.threshold,
      left: this.buildTree(partition.leftIndices, depth + 1),
      right: this.buildTree(partition.rightIndices, depth + 1),
    };
  }

  private resolveMaxFeatures(featureCount: number): number {
    if (this.maxFeatures === null || this.maxFeatures === undefined) {
      return featureCount;
    }
    if (this.maxFeatures === "sqrt") {
      return Math.max(1, Math.floor(Math.sqrt(featureCount)));
    }
    if (this.maxFeatures === "log2") {
      return Math.max(1, Math.floor(Math.log2(featureCount)));
    }
    return Math.max(1, Math.min(featureCount, Math.floor(this.maxFeatures)));
  }

  private selectFeatureIndices(featureCount: number): number[] {
    const k = this.resolveMaxFeatures(featureCount);
    if (k >= featureCount) {
      return this.allFeatureIndices;
    }

    const marks = this.featureSelectionMarks!;
    marks.fill(0);
    const selected = new Array<number>(k);
    let count = 0;
    while (count < k) {
      const candidate = Math.floor(this.random() * featureCount);
      if (marks[candidate] !== 0) {
        continue;
      }
      marks[candidate] = 1;
      selected[count] = candidate;
      count += 1;
    }
    return selected;
  }

  private findBestThreshold(indices: number[], featureIndex: number): SplitEvaluation | null {
    const x = this.flattenedXTrain!;
    const y = this.yTrain!;
    const stride = this.featureCount;
    const sampleCount = indices.length;

    let minValue = Number.POSITIVE_INFINITY;
    let maxValue = Number.NEGATIVE_INFINITY;
    let totalSum = 0;
    let totalSumSquares = 0;

    for (let i = 0; i < sampleCount; i += 1) {
      const sampleIndex = indices[i];
      const xValue = x[sampleIndex * stride + featureIndex];
      const yValue = y[sampleIndex];
      if (xValue < minValue) {
        minValue = xValue;
      }
      if (xValue > maxValue) {
        maxValue = xValue;
      }
      totalSum += yValue;
      totalSumSquares += yValue * yValue;
    }

    if (!Number.isFinite(minValue) || !Number.isFinite(maxValue) || minValue === maxValue) {
      return null;
    }
    if (this.splitter === "random") {
      const threshold = minValue + this.random() * (maxValue - minValue);
      let leftCount = 0;
      let leftSum = 0;
      let leftSumSquares = 0;
      for (let i = 0; i < sampleCount; i += 1) {
        const sampleIndex = indices[i];
        if (x[sampleIndex * stride + featureIndex] <= threshold) {
          const target = y[sampleIndex];
          leftCount += 1;
          leftSum += target;
          leftSumSquares += target * target;
        }
      }

      const rightCount = sampleCount - leftCount;
      if (leftCount < this.minSamplesLeaf || rightCount < this.minSamplesLeaf) {
        return null;
      }
      const rightSum = totalSum - leftSum;
      const rightSumSquares = totalSumSquares - leftSumSquares;
      const leftVariance = safeVariance(leftSum, leftSumSquares, leftCount);
      const rightVariance = safeVariance(rightSum, rightSumSquares, rightCount);
      const impurity =
        (leftCount / sampleCount) * leftVariance + (rightCount / sampleCount) * rightVariance;
      return { threshold, impurity };
    }

    const dynamicBins = Math.floor(Math.sqrt(sampleCount));
    const binCount = Math.max(16, Math.min(MAX_THRESHOLD_BINS, dynamicBins));
    const binTotals = this.binTotals;
    const binSums = this.binSums;
    const binSumsSquares = this.binSumsSquares;
    binTotals.fill(0, 0, binCount);
    binSums.fill(0, 0, binCount);
    binSumsSquares.fill(0, 0, binCount);

    const range = maxValue - minValue;
    for (let i = 0; i < sampleCount; i += 1) {
      const sampleIndex = indices[i];
      const xValue = x[sampleIndex * stride + featureIndex];
      const yValue = y[sampleIndex];
      let bin = Math.floor(((xValue - minValue) / range) * binCount);
      if (bin < 0) {
        bin = 0;
      } else if (bin >= binCount) {
        bin = binCount - 1;
      }
      binTotals[bin] += 1;
      binSums[bin] += yValue;
      binSumsSquares[bin] += yValue * yValue;
    }

    let leftCount = 0;
    let leftSum = 0;
    let leftSumSquares = 0;
    let bestImpurity = Number.POSITIVE_INFINITY;
    let bestThreshold = 0;

    for (let bin = 0; bin < binCount - 1; bin += 1) {
      leftCount += binTotals[bin];
      leftSum += binSums[bin];
      leftSumSquares += binSumsSquares[bin];

      const rightCount = sampleCount - leftCount;
      if (leftCount < this.minSamplesLeaf || rightCount < this.minSamplesLeaf) {
        continue;
      }

      const rightSum = totalSum - leftSum;
      const rightSumSquares = totalSumSquares - leftSumSquares;

      const leftVariance = safeVariance(leftSum, leftSumSquares, leftCount);
      const rightVariance = safeVariance(rightSum, rightSumSquares, rightCount);
      const impurity =
        (leftCount / sampleCount) * leftVariance + (rightCount / sampleCount) * rightVariance;

      if (impurity < bestImpurity) {
        bestImpurity = impurity;
        bestThreshold = minValue + (range * (bin + 1)) / binCount;
      }
    }

    if (!Number.isFinite(bestImpurity)) {
      return null;
    }

    return {
      threshold: bestThreshold,
      impurity: bestImpurity,
    };
  }

  private partitionIndices(
    indices: number[],
    featureIndex: number,
    threshold: number,
  ): SplitPartition | null {
    const x = this.flattenedXTrain!;
    const stride = this.featureCount;
    const sampleCount = indices.length;
    const leftIndices = new Array<number>(sampleCount);
    const rightIndices = new Array<number>(sampleCount);
    let leftCount = 0;
    let rightCount = 0;

    for (let i = 0; i < sampleCount; i += 1) {
      const sampleIndex = indices[i];
      if (x[sampleIndex * stride + featureIndex] <= threshold) {
        leftIndices[leftCount] = sampleIndex;
        leftCount += 1;
      } else {
        rightIndices[rightCount] = sampleIndex;
        rightCount += 1;
      }
    }

    if (leftCount < this.minSamplesLeaf || rightCount < this.minSamplesLeaf) {
      return null;
    }

    return {
      leftIndices: leftIndices.slice(0, leftCount),
      rightIndices: rightIndices.slice(0, rightCount),
    };
  }

  private flattenTrainingMatrix(X: Matrix): Float64Array {
    const sampleCount = X.length;
    const flattened = new Float64Array(sampleCount * this.featureCount);
    for (let i = 0; i < sampleCount; i += 1) {
      const row = X[i];
      const rowOffset = i * this.featureCount;
      for (let j = 0; j < this.featureCount; j += 1) {
        flattened[rowOffset + j] = row[j];
      }
    }
    return flattened;
  }

  private toFloat64Vector(y: Vector): Float64Array {
    const out = new Float64Array(y.length);
    for (let i = 0; i < y.length; i += 1) {
      out[i] = y[i];
    }
    return out;
  }

  private finalizeFeatureImportances(): void {
    if (!this.featureImportanceRaw) {
      this.featureImportances_ = null;
      return;
    }
    const out = new Array<number>(this.featureImportanceRaw.length);
    let total = 0;
    for (let i = 0; i < this.featureImportanceRaw.length; i += 1) {
      total += this.featureImportanceRaw[i];
    }
    if (total <= 0) {
      for (let i = 0; i < out.length; i += 1) {
        out[i] = 0;
      }
      this.featureImportances_ = out;
      return;
    }
    for (let i = 0; i < out.length; i += 1) {
      out[i] = this.featureImportanceRaw[i] / total;
    }
    this.featureImportances_ = out;
  }
}
