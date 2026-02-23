import type { ClassificationModel, Matrix, Vector } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  validateClassificationInputs,
} from "../utils/validation";
import { accuracyScore } from "../metrics/classification";
import { getZigKernels } from "../native/zigKernels";

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

interface SplitEvaluation {
  threshold: number;
  impurity: number;
}

interface SplitPartition {
  leftIndices: number[];
  rightIndices: number[];
}

const MAX_THRESHOLD_BINS = 128;

function isZigTreeBackendEnabled(): boolean {
  const mode = process.env.BUN_SCIKIT_TREE_BACKEND?.trim().toLowerCase();
  return mode === "zig" || mode === "native";
}

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
  fitBackend_: "zig" | "js" = "js";
  fitBackendLibrary_: string | null = null;
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
  private binTotals: Uint32Array = new Uint32Array(MAX_THRESHOLD_BINS);
  private binPositives: Uint32Array = new Uint32Array(MAX_THRESHOLD_BINS);
  private zigModelHandle: bigint | null = null;

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
    this.destroyZigModel();

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

    let validatedSampleIndices: Uint32Array | null = null;
    if (sampleIndices) {
      if (sampleIndices.length === 0) {
        throw new Error("sampleIndices must not be empty.");
      }
      validatedSampleIndices = new Uint32Array(sampleIndices.length);
      for (let i = 0; i < sampleIndices.length; i += 1) {
        const index = sampleIndices[i];
        if (!Number.isInteger(index) || index < 0 || index >= X.length) {
          throw new Error(`sampleIndices contains invalid index: ${index}.`);
        }
        validatedSampleIndices[i] = index;
      }
    }

    if (isZigTreeBackendEnabled() && this.tryFitWithZig(X.length, validatedSampleIndices)) {
      return this;
    }

    let rootIndices: number[];
    if (validatedSampleIndices) {
      rootIndices = Array.from(validatedSampleIndices);
    } else {
      rootIndices = new Array<number>(X.length);
      for (let idx = 0; idx < X.length; idx += 1) {
        rootIndices[idx] = idx;
      }
    }

    this.root = this.buildTree(rootIndices, 0);
    this.fitBackend_ = "js";
    this.fitBackendLibrary_ = null;
    return this;
  }

  predict(X: Matrix): Vector {
    if ((this.root === null && this.zigModelHandle === null) || this.featureCount === 0) {
      throw new Error("DecisionTreeClassifier has not been fitted.");
    }

    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    if (X[0].length !== this.featureCount) {
      throw new Error(
        `Feature size mismatch. Expected ${this.featureCount}, got ${X[0].length}.`,
      );
    }

    if (this.zigModelHandle !== null) {
      const kernels = getZigKernels();
      const nativePredict = kernels?.decisionTreeModelPredict;
      if (nativePredict) {
        const flattenedX = this.flattenTrainingMatrix(X);
        const outLabels = new Uint8Array(X.length);
        const status = nativePredict(
          this.zigModelHandle,
          flattenedX,
          X.length,
          this.featureCount,
          outLabels,
        );
        if (status === 1) {
          return Array.from(outLabels);
        }
      }
      if (!this.root) {
        throw new Error("Native DecisionTree predict failed and no JS fallback tree is available.");
      }
    }

    return X.map((sample) => this.predictOne(sample, this.root!));
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return accuracyScore(y, this.predict(X));
  }

  private predictOne(sample: Vector, node: TreeNode): 0 | 1 {
    let current: TreeNode = node;
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
    let bestSplit: SplitEvaluation | null = null;

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

    const partition = this.partitionIndices(indices, bestFeature, bestSplit.threshold);
    if (!partition) {
      return { isLeaf: true, prediction };
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
    if (!Number.isFinite(this.maxFeatures)) {
      return featureCount;
    }
    return Math.max(1, Math.min(featureCount, Math.floor(this.maxFeatures)));
  }

  private resolveNativeMaxFeatures(featureCount: number): {
    mode: 0 | 1 | 2 | 3;
    value: number;
  } {
    if (this.maxFeatures === null || this.maxFeatures === undefined) {
      return { mode: 0, value: 0 };
    }
    if (this.maxFeatures === "sqrt") {
      return { mode: 1, value: 0 };
    }
    if (this.maxFeatures === "log2") {
      return { mode: 2, value: 0 };
    }
    const value = Number.isFinite(this.maxFeatures)
      ? Math.max(1, Math.min(featureCount, Math.floor(this.maxFeatures)))
      : featureCount;
    return { mode: 3, value };
  }

  private tryFitWithZig(
    sampleCount: number,
    sampleIndices: Uint32Array | null,
  ): boolean {
    const kernels = getZigKernels();
    const create = kernels?.decisionTreeModelCreate;
    const fit = kernels?.decisionTreeModelFit;
    const destroy = kernels?.decisionTreeModelDestroy;
    if (!create || !fit || !destroy) {
      return false;
    }

    const { mode, value } = this.resolveNativeMaxFeatures(this.featureCount);
    const useRandomState = this.randomState === undefined ? 0 : 1;
    const randomState = this.randomState ?? 0;
    const handle = create(
      this.maxDepth,
      this.minSamplesSplit,
      this.minSamplesLeaf,
      mode,
      value,
      randomState >>> 0,
      useRandomState,
      this.featureCount,
    );
    if (handle === 0n) {
      return false;
    }

    let shouldDestroy = true;
    try {
      const emptySampleIndices = new Uint32Array(0);
      const status = fit(
        handle,
        this.flattenedXTrain!,
        this.yBinaryTrain!,
        sampleCount,
        this.featureCount,
        sampleIndices ?? emptySampleIndices,
        sampleIndices?.length ?? 0,
      );
      if (status !== 1) {
        return false;
      }

      this.zigModelHandle = handle;
      this.root = null;
      this.fitBackend_ = "zig";
      this.fitBackendLibrary_ = kernels.libraryPath;
      shouldDestroy = false;
      return true;
    } catch {
      return false;
    } finally {
      if (shouldDestroy) {
        destroy(handle);
      }
    }
  }

  private destroyZigModel(): void {
    if (this.zigModelHandle === null) {
      return;
    }
    const kernels = getZigKernels();
    const destroy = kernels?.decisionTreeModelDestroy;
    if (destroy) {
      try {
        destroy(this.zigModelHandle);
      } catch {
        // no-op: cleanup best effort
      }
    }
    this.zigModelHandle = null;
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

  private findBestThreshold(indices: number[], featureIndex: number): SplitEvaluation | null {
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
    const binTotals = this.binTotals;
    const binPositives = this.binPositives;
    binTotals.fill(0, 0, binCount);
    binPositives.fill(0, 0, binCount);
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
    let leftPartitionCount = 0;
    let rightPartitionCount = 0;
    for (let i = 0; i < sampleCount; i += 1) {
      const sampleIndex = indices[i];
      if (x[sampleIndex * stride + featureIndex] <= threshold) {
        leftIndices[leftPartitionCount] = sampleIndex;
        leftPartitionCount += 1;
      } else {
        rightIndices[rightPartitionCount] = sampleIndex;
        rightPartitionCount += 1;
      }
    }

    if (
      leftPartitionCount < this.minSamplesLeaf ||
      rightPartitionCount < this.minSamplesLeaf
    ) {
      return null;
    }

    return {
      leftIndices: leftIndices.slice(0, leftPartitionCount),
      rightIndices: rightIndices.slice(0, rightPartitionCount),
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
