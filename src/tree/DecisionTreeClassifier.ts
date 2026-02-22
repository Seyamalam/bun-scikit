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

function majorityClass(y: Vector): 0 | 1 {
  let positives = 0;
  for (let i = 0; i < y.length; i += 1) {
    positives += y[i] === 1 ? 1 : 0;
  }
  return positives * 2 >= y.length ? 1 : 0;
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

  constructor(options: DecisionTreeClassifierOptions = {}) {
    this.maxDepth = options.maxDepth ?? 12;
    this.minSamplesSplit = options.minSamplesSplit ?? 2;
    this.minSamplesLeaf = options.minSamplesLeaf ?? 1;
    this.maxFeatures = options.maxFeatures ?? null;
    this.randomState = options.randomState;
  }

  fit(X: Matrix, y: Vector): this {
    validateClassificationInputs(X, y);

    this.random = this.randomState === undefined ? Math.random : mulberry32(this.randomState);
    this.root = this.buildTree(X, y, 0);
    return this;
  }

  predict(X: Matrix): Vector {
    if (!this.root) {
      throw new Error("DecisionTreeClassifier has not been fitted.");
    }

    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

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

  private buildTree(X: Matrix, y: Vector, depth: number): TreeNode {
    const sampleCount = X.length;
    const prediction = majorityClass(y);
    const positiveCount = y.reduce((sum, value) => sum + (value === 1 ? 1 : 0), 0);

    const sameClass = positiveCount === 0 || positiveCount === sampleCount;
    const depthStop = depth >= this.maxDepth;
    const splitStop = sampleCount < this.minSamplesSplit;

    if (sameClass || depthStop || splitStop) {
      return { isLeaf: true, prediction };
    }

    const featureCount = X[0].length;
    const candidateFeatures = this.selectFeatureIndices(featureCount);
    const parentImpurity = giniImpurity(positiveCount, sampleCount);

    let bestFeature = -1;
    let bestThreshold = 0;
    let bestImpurity = Number.POSITIVE_INFINITY;
    let bestLeftIndices: number[] = [];
    let bestRightIndices: number[] = [];

    for (let idx = 0; idx < candidateFeatures.length; idx += 1) {
      const featureIndex = candidateFeatures[idx];
      const split = this.findBestThreshold(X, y, featureIndex);
      if (!split) {
        continue;
      }

      if (split.impurity < bestImpurity) {
        bestImpurity = split.impurity;
        bestFeature = featureIndex;
        bestThreshold = split.threshold;
        bestLeftIndices = split.leftIndices;
        bestRightIndices = split.rightIndices;
      }
    }

    if (bestFeature === -1 || bestImpurity >= parentImpurity - 1e-12) {
      return { isLeaf: true, prediction };
    }

    const leftX = bestLeftIndices.map((i) => X[i]);
    const leftY = bestLeftIndices.map((i) => y[i]);
    const rightX = bestRightIndices.map((i) => X[i]);
    const rightY = bestRightIndices.map((i) => y[i]);

    return {
      isLeaf: false,
      prediction,
      featureIndex: bestFeature,
      threshold: bestThreshold,
      left: this.buildTree(leftX, leftY, depth + 1),
      right: this.buildTree(rightX, rightY, depth + 1),
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
    const features = Array.from({ length: featureCount }, (_, i) => i);
    if (k >= featureCount) {
      return features;
    }

    for (let i = features.length - 1; i > 0; i -= 1) {
      const j = Math.floor(this.random() * (i + 1));
      const tmp = features[i];
      features[i] = features[j];
      features[j] = tmp;
    }

    return features.slice(0, k);
  }

  private findBestThreshold(
    X: Matrix,
    y: Vector,
    featureIndex: number,
  ): {
    threshold: number;
    impurity: number;
    leftIndices: number[];
    rightIndices: number[];
  } | null {
    const sampleCount = X.length;
    const rows = Array.from({ length: sampleCount }, (_, idx) => ({
      value: X[idx][featureIndex],
      label: y[idx],
      index: idx,
    })).sort((a, b) => a.value - b.value);

    let totalPositive = 0;
    for (let i = 0; i < rows.length; i += 1) {
      totalPositive += rows[i].label === 1 ? 1 : 0;
    }

    let leftCount = 0;
    let leftPositive = 0;
    let bestImpurity = Number.POSITIVE_INFINITY;
    let bestThreshold = 0;

    for (let i = 1; i < rows.length; i += 1) {
      leftCount += 1;
      leftPositive += rows[i - 1].label === 1 ? 1 : 0;
      const rightCount = sampleCount - leftCount;

      if (leftCount < this.minSamplesLeaf || rightCount < this.minSamplesLeaf) {
        continue;
      }

      const leftValue = rows[i - 1].value;
      const rightValue = rows[i].value;
      if (leftValue === rightValue) {
        continue;
      }

      const rightPositive = totalPositive - leftPositive;
      const impurity =
        (leftCount / sampleCount) * giniImpurity(leftPositive, leftCount) +
        (rightCount / sampleCount) * giniImpurity(rightPositive, rightCount);

      if (impurity < bestImpurity) {
        bestImpurity = impurity;
        bestThreshold = (leftValue + rightValue) / 2;
      }
    }

    if (!Number.isFinite(bestImpurity)) {
      return null;
    }

    const leftIndices: number[] = [];
    const rightIndices: number[] = [];
    for (let i = 0; i < sampleCount; i += 1) {
      if (X[i][featureIndex] <= bestThreshold) {
        leftIndices.push(i);
      } else {
        rightIndices.push(i);
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
}
