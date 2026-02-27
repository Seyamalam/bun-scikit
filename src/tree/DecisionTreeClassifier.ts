import type { ClassificationModel, Matrix, Vector } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  validateClassificationInputs,
} from "../utils/validation";
import { accuracyScore } from "../metrics/classification";
import { getZigKernels } from "../native/zigKernels";
import { buildLabelIndex, uniqueSortedLabels } from "../utils/classification";

export type MaxFeaturesOption = "sqrt" | "log2" | number | null;

export interface DecisionTreeClassifierOptions {
  maxDepth?: number;
  minSamplesSplit?: number;
  minSamplesLeaf?: number;
  maxFeatures?: MaxFeaturesOption;
  randomState?: number;
}

interface TreeNode {
  predictionClassIndex: number;
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
  if (!mode || mode === "auto" || mode === "zig" || mode === "native") {
    return true;
  }
  if (mode === "js" || mode === "off" || mode === "false" || mode === "0" || mode === "ts") {
    return false;
  }
  return true;
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

function giniImpurity(counts: Uint32Array | number[], sampleCount: number): number {
  if (sampleCount === 0) {
    return 0;
  }
  let sumSquares = 0;
  for (let i = 0; i < counts.length; i += 1) {
    const p = counts[i] / sampleCount;
    sumSquares += p * p;
  }
  return 1 - sumSquares;
}

function majorityClassIndex(counts: Uint32Array | number[]): number {
  let best = 0;
  let bestCount = counts[0];
  for (let i = 1; i < counts.length; i += 1) {
    if (counts[i] > bestCount) {
      bestCount = counts[i];
      best = i;
    }
  }
  return best;
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
  private yEncodedTrain: Uint16Array | null = null;
  private yBinaryTrain: Uint8Array | null = null;
  private featureCount = 0;
  private classCount = 0;
  private allFeatureIndices: number[] = [];
  private featureSelectionMarks: Uint8Array | null = null;
  private binTotals: Uint32Array = new Uint32Array(MAX_THRESHOLD_BINS);
  private binClassCounts: Uint32Array = new Uint32Array(MAX_THRESHOLD_BINS);
  private zigModelHandle: bigint | null = null;
  private nativeBinaryEligible = false;

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
    yEncodedTrain?: Uint16Array,
    classes?: Vector,
  ): this {
    this.destroyZigModel();

    if (!skipValidation) {
      validateClassificationInputs(X, y);
    }
    this.featureCount = X[0].length;
    this.classes_ = classes ? classes.slice() : uniqueSortedLabels(y);
    this.classCount = this.classes_.length;
    if (this.classCount < 2) {
      throw new Error("DecisionTreeClassifier requires at least two classes.");
    }
    const classToIndex = buildLabelIndex(this.classes_);

    this.flattenedXTrain = flattenedXTrain ?? this.flattenTrainingMatrix(X);
    this.yEncodedTrain =
      yEncodedTrain ?? this.encodeTargets(y, classToIndex);
    this.nativeBinaryEligible =
      this.classes_.length === 2 && this.classes_[0] === 0 && this.classes_[1] === 1;
    this.yBinaryTrain = this.nativeBinaryEligible
      ? this.buildBinaryTargetsFromEncoded(this.yEncodedTrain)
      : null;

    this.allFeatureIndices = new Array<number>(this.featureCount);
    for (let i = 0; i < this.featureCount; i += 1) {
      this.allFeatureIndices[i] = i;
    }
    this.featureSelectionMarks = new Uint8Array(this.featureCount);
    this.random = this.randomState === undefined ? Math.random : mulberry32(this.randomState);
    this.binClassCounts = new Uint32Array(MAX_THRESHOLD_BINS * this.classCount);

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

    if (
      this.nativeBinaryEligible &&
      isZigTreeBackendEnabled() &&
      this.tryFitWithZig(X.length, validatedSampleIndices)
    ) {
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
          return Array.from(outLabels).map((idx) => this.classes_[idx]);
        }
      }
      if (!this.root) {
        throw new Error("Native DecisionTree predict failed and no JS fallback tree is available.");
      }
    }

    const predictions = new Array<number>(X.length);
    const root = this.root!;
    for (let i = 0; i < X.length; i += 1) {
      predictions[i] = this.predictOne(X[i], root);
    }
    return predictions;
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return accuracyScore(y, this.predict(X));
  }

  dispose(): void {
    this.destroyZigModel();
    this.root = null;
    this.flattenedXTrain = null;
    this.yEncodedTrain = null;
    this.yBinaryTrain = null;
  }

  private predictOne(sample: Vector, node: TreeNode): number {
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
    return this.classes_[current.predictionClassIndex];
  }

  private buildTree(indices: number[], depth: number): TreeNode {
    const y = this.yEncodedTrain!;
    const sampleCount = indices.length;
    const classCounts = new Uint32Array(this.classCount);
    for (let i = 0; i < sampleCount; i += 1) {
      classCounts[y[indices[i]]] += 1;
    }
    const predictionClassIndex = majorityClassIndex(classCounts);

    const maxClassCount = classCounts[predictionClassIndex];
    const sameClass = maxClassCount === sampleCount;
    const depthStop = depth >= this.maxDepth;
    const splitStop = sampleCount < this.minSamplesSplit;
    if (sameClass || depthStop || splitStop) {
      return { isLeaf: true, predictionClassIndex };
    }

    const candidateFeatures = this.selectFeatureIndices(this.featureCount);
    const parentImpurity = giniImpurity(classCounts, sampleCount);

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
      return { isLeaf: true, predictionClassIndex };
    }

    const partition = this.partitionIndices(indices, bestFeature, bestSplit.threshold);
    if (!partition) {
      return { isLeaf: true, predictionClassIndex };
    }

    return {
      isLeaf: false,
      predictionClassIndex,
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
    if (!create || !fit || !destroy || !this.yBinaryTrain) {
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
        this.yBinaryTrain,
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
    const y = this.yEncodedTrain!;
    const stride = this.featureCount;
    const sampleCount = indices.length;
    let minValue = Number.POSITIVE_INFINITY;
    let maxValue = Number.NEGATIVE_INFINITY;
    const totalClassCounts = new Uint32Array(this.classCount);

    for (let i = 0; i < sampleCount; i += 1) {
      const sampleIndex = indices[i];
      const value = x[sampleIndex * stride + featureIndex];
      if (value < minValue) {
        minValue = value;
      }
      if (value > maxValue) {
        maxValue = value;
      }
      totalClassCounts[y[sampleIndex]] += 1;
    }

    if (!Number.isFinite(minValue) || !Number.isFinite(maxValue) || minValue === maxValue) {
      return null;
    }

    const dynamicBins = Math.floor(Math.sqrt(sampleCount));
    const binCount = Math.max(16, Math.min(MAX_THRESHOLD_BINS, dynamicBins));
    const binTotals = this.binTotals;
    const binClassCounts = this.binClassCounts;
    binTotals.fill(0, 0, binCount);
    binClassCounts.fill(0, 0, binCount * this.classCount);
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
      const classIndex = y[sampleIndex];
      binTotals[bin] += 1;
      binClassCounts[bin * this.classCount + classIndex] += 1;
    }

    const leftClassCounts = new Uint32Array(this.classCount);
    let leftCount = 0;
    let bestImpurity = Number.POSITIVE_INFINITY;
    let bestThreshold = 0;

    for (let bin = 0; bin < binCount - 1; bin += 1) {
      leftCount += binTotals[bin];
      const rightCount = sampleCount - leftCount;
      for (let classIndex = 0; classIndex < this.classCount; classIndex += 1) {
        leftClassCounts[classIndex] += binClassCounts[bin * this.classCount + classIndex];
      }

      if (leftCount < this.minSamplesLeaf || rightCount < this.minSamplesLeaf) {
        continue;
      }

      let leftGiniSum = 0;
      let rightGiniSum = 0;
      for (let classIndex = 0; classIndex < this.classCount; classIndex += 1) {
        const leftClassCount = leftClassCounts[classIndex];
        const rightClassCount = totalClassCounts[classIndex] - leftClassCount;
        const lp = leftClassCount / leftCount;
        const rp = rightClassCount / rightCount;
        leftGiniSum += lp * lp;
        rightGiniSum += rp * rp;
      }
      const impurity =
        (leftCount / sampleCount) * (1 - leftGiniSum) +
        (rightCount / sampleCount) * (1 - rightGiniSum);

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

  private encodeTargets(y: Vector, classToIndex: Map<number, number>): Uint16Array {
    const encoded = new Uint16Array(y.length);
    for (let i = 0; i < y.length; i += 1) {
      const classIndex = classToIndex.get(y[i]);
      if (classIndex === undefined) {
        throw new Error(`Unknown class label '${y[i]}' in target vector.`);
      }
      encoded[i] = classIndex;
    }
    return encoded;
  }

  private buildBinaryTargetsFromEncoded(yEncoded: Uint16Array): Uint8Array {
    const encoded = new Uint8Array(yEncoded.length);
    for (let i = 0; i < yEncoded.length; i += 1) {
      encoded[i] = yEncoded[i] === 1 ? 1 : 0;
    }
    return encoded;
  }
}
