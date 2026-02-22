import type { ClassificationModel, Matrix, Vector } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  validateClassificationInputs,
} from "../utils/validation";
import { accuracyScore } from "../metrics/classification";

export type MaxFeaturesOption = "sqrt" | "log2" | number | null;

export interface DecisionTreeClassifierOptions {
  maxDepth?: number;
  minSamplesSplit?: number;
  minSamplesLeaf?: number;
  maxFeatures?: MaxFeaturesOption;
  randomState?: number;
}

interface TreeNode {
  prediction: 0 | 1;
  featureIndex?: number;
  threshold?: number;
  left?: TreeNode;
  right?: TreeNode;
  isLeaf: boolean;
}

interface SplitCandidate {
  threshold: number;
  impurity: number;
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

function giniImpurity(positiveCount: number, sampleCount: number): number {
  if (sampleCount === 0) {
    return 0;
  }
  const p1 = positiveCount / sampleCount;
  const p0 = 1 - p1;
  return 1 - p1 * p1 - p0 * p0;
}

export class DecisionTreeClassifier implements ClassificationModel {
  classes_: Vector = [0, 1];
  private readonly maxDepth: number;
  private readonly minSamplesSplit: number;
  private readonly minSamplesLeaf: number;
  private readonly maxFeatures: MaxFeaturesOption;
  private readonly randomState?: number;
  private random: () => number = Math.random;
  private root: TreeNode | null = null;
  private flattenedXTrain: Float64Array | null = null;
  private yBinaryTrain: Uint8Array | null = null;
  private featureCount = 0;
  private allFeatureIndices: number[] = [];
  private featureSelectionMarks: Uint8Array | null = null;

  constructor(options: DecisionTreeClassifierOptions = {}) {
    this.maxDepth = options.maxDepth ?? 12;
    this.minSamplesSplit = options.minSamplesSplit ?? 2;
    this.minSamplesLeaf = options.minSamplesLeaf ?? 1;
    this.maxFeatures = options.maxFeatures ?? null;
    this.randomState = options.randomState;
  }

  fit(
    X: Matrix,
    y: Vector,
    sampleIndices?: ArrayLike<number>,
    skipValidation = false,
    flattenedXTrain?: Float64Array,
    yBinaryTrain?: Uint8Array,
  ): this {
    if (!skipValidation) {
      validateClassificationInputs(X, y);
    }
    this.featureCount = X[0].length;
    this.flattenedXTrain = flattenedXTrain ?? this.flattenTrainingMatrix(X);
    this.yBinaryTrain = yBinaryTrain ?? this.buildBinaryTargets(y);
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
      for (let i = 0; i < sampleIndices.length; i += 1) {
        const index = sampleIndices[i];
        if (!Number.isInteger(index) || index < 0 || index >= X.length) {
          throw new Error(`sampleIndices contains invalid index: ${index}.`);
        }
      }
      rootIndices = Array.from(sampleIndices);
    } else {
      rootIndices = new Array<number>(X.length);
      for (let idx = 0; idx < X.length; idx += 1) {
        rootIndices[idx] = idx;
      }
    }

    this.root = this.buildTree(rootIndices, 0);
    return this;
  }

  predict(X: Matrix): Vector {
    if (!this.root || this.featureCount === 0) {
      throw new Error("DecisionTreeClassifier has not been fitted.");
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
    return accuracyScore(y, this.predict(X));
  }

  private predictOne(sample: Vector, node: TreeNode): 0 | 1 {
    if (node.isLeaf || node.featureIndex === undefined || node.threshold === undefined) {
      return node.prediction;
    }

    if (sample[node.featureIndex] <= node.threshold) {
      return this.predictOne(sample, node.left!);
    }
    return this.predictOne(sample, node.right!);
  }

  private buildTree(indices: number[], depth: number): TreeNode {
    const y = this.yBinaryTrain!;
    const sampleCount = indices.length;
    let positiveCount = 0;
    for (let i = 0; i < sampleCount; i += 1) {
      positiveCount += y[indices[i]];
    }
    const prediction: 0 | 1 = positiveCount * 2 >= sampleCount ? 1 : 0;

    const sameClass = positiveCount === 0 || positiveCount === sampleCount;
    const depthStop = depth >= this.maxDepth;
    const splitStop = sampleCount < this.minSamplesSplit;
    if (sameClass || depthStop || splitStop) {
      return { isLeaf: true, prediction };
    }

    const candidateFeatures = this.selectFeatureIndices(this.featureCount);
    const parentImpurity = giniImpurity(positiveCount, sampleCount);

    let bestFeature = -1;
    let bestSplit: SplitCandidate | null = null;

    for (let idx = 0; idx < candidateFeatures.length; idx += 1) {
      const featureIndex = candidateFeatures[idx];
      const split = this.findBestThreshold(indices, featureIndex);
      if (!split) {
        continue;
      }

      if (!bestSplit || split.impurity < bestSplit.impurity) {
        bestFeature = featureIndex;
        bestSplit = split;
      }
    }

    if (!bestSplit || bestFeature === -1 || bestSplit.impurity >= parentImpurity - 1e-12) {
      return { isLeaf: true, prediction };
    }

    return {
      isLeaf: false,
      prediction,
      featureIndex: bestFeature,
      threshold: bestSplit.threshold,
      left: this.buildTree(bestSplit.leftIndices, depth + 1),
      right: this.buildTree(bestSplit.rightIndices, depth + 1),
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
    let selectedCount = 0;
    while (selectedCount < k) {
      const candidate = Math.floor(this.random() * featureCount);
      if (marks[candidate] !== 0) {
        continue;
      }
      marks[candidate] = 1;
      selected[selectedCount] = candidate;
      selectedCount += 1;
    }

    return selected;
  }

  private findBestThreshold(indices: number[], featureIndex: number): SplitCandidate | null {
    const x = this.flattenedXTrain!;
    const y = this.yBinaryTrain!;
    const stride = this.featureCount;
    const sampleCount = indices.length;
    let minValue = Number.POSITIVE_INFINITY;
    let maxValue = Number.NEGATIVE_INFINITY;
    let totalPositive = 0;
    for (let i = 0; i < sampleCount; i += 1) {
      const sampleIndex = indices[i];
      const value = x[sampleIndex * stride + featureIndex];
      if (value < minValue) {
        minValue = value;
      }
      if (value > maxValue) {
        maxValue = value;
      }
      totalPositive += y[sampleIndex];
    }

    if (!Number.isFinite(minValue) || !Number.isFinite(maxValue) || minValue === maxValue) {
      return null;
    }

    const dynamicBins = Math.floor(Math.sqrt(sampleCount));
    const binCount = Math.max(16, Math.min(MAX_THRESHOLD_BINS, dynamicBins));
    const binTotals = new Uint32Array(binCount);
    const binPositives = new Uint32Array(binCount);
    const range = maxValue - minValue;

    for (let i = 0; i < sampleCount; i += 1) {
      const sampleIndex = indices[i];
      const value = x[sampleIndex * stride + featureIndex];
      let bin = Math.floor(((value - minValue) / range) * binCount);
      if (bin < 0) {
        bin = 0;
      } else if (bin >= binCount) {
        bin = binCount - 1;
      }
      binTotals[bin] += 1;
      binPositives[bin] += y[sampleIndex];
    }

    let leftCount = 0;
    let leftPositive = 0;
    let bestImpurity = Number.POSITIVE_INFINITY;
    let bestThreshold = 0;

    for (let bin = 0; bin < binCount - 1; bin += 1) {
      leftCount += binTotals[bin];
      leftPositive += binPositives[bin];
      const rightCount = sampleCount - leftCount;

      if (leftCount < this.minSamplesLeaf || rightCount < this.minSamplesLeaf) {
        continue;
      }

      const rightPositive = totalPositive - leftPositive;
      const impurity =
        (leftCount / sampleCount) * giniImpurity(leftPositive, leftCount) +
        (rightCount / sampleCount) * giniImpurity(rightPositive, rightCount);

      if (impurity < bestImpurity) {
        bestImpurity = impurity;
        bestThreshold = minValue + (range * (bin + 1)) / binCount;
      }
    }

    if (!Number.isFinite(bestImpurity)) {
      return null;
    }

    const leftIndices: number[] = [];
    const rightIndices: number[] = [];
    for (let i = 0; i < sampleCount; i += 1) {
      const sampleIndex = indices[i];
      if (x[sampleIndex * stride + featureIndex] <= bestThreshold) {
        leftIndices.push(sampleIndex);
      } else {
        rightIndices.push(sampleIndex);
      }
    }

    if (
      leftIndices.length < this.minSamplesLeaf ||
      rightIndices.length < this.minSamplesLeaf
    ) {
      return null;
    }

    return {
      threshold: bestThreshold,
      impurity: bestImpurity,
      leftIndices,
      rightIndices,
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

  private buildBinaryTargets(y: Vector): Uint8Array {
    const encoded = new Uint8Array(y.length);
    for (let i = 0; i < y.length; i += 1) {
      encoded[i] = y[i] === 1 ? 1 : 0;
    }
    return encoded;
  }
}
