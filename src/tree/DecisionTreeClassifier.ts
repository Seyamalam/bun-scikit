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
  private XTrain: Matrix | null = null;
  private yTrain: Vector | null = null;

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
    sampleIndices?: number[],
    skipValidation = false,
  ): this {
    if (!skipValidation) {
      validateClassificationInputs(X, y);
    }
    this.XTrain = X;
    this.yTrain = y;
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
      rootIndices = [...sampleIndices];
    } else {
      rootIndices = Array.from({ length: X.length }, (_, idx) => idx);
    }

    this.root = this.buildTree(rootIndices, 0);
    return this;
  }

  predict(X: Matrix): Vector {
    if (!this.root || !this.XTrain) {
      throw new Error("DecisionTreeClassifier has not been fitted.");
    }

    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    if (X[0].length !== this.XTrain[0].length) {
      throw new Error(
        `Feature size mismatch. Expected ${this.XTrain[0].length}, got ${X[0].length}.`,
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
    const y = this.yTrain!;
    const sampleCount = indices.length;
    let positiveCount = 0;
    for (let i = 0; i < sampleCount; i += 1) {
      positiveCount += y[indices[i]] === 1 ? 1 : 0;
    }
    const prediction: 0 | 1 = positiveCount * 2 >= sampleCount ? 1 : 0;

    const sameClass = positiveCount === 0 || positiveCount === sampleCount;
    const depthStop = depth >= this.maxDepth;
    const splitStop = sampleCount < this.minSamplesSplit;
    if (sameClass || depthStop || splitStop) {
      return { isLeaf: true, prediction };
    }

    const featureCount = this.XTrain![0].length;
    const candidateFeatures = this.selectFeatureIndices(featureCount);
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
    const features = Array.from({ length: featureCount }, (_, i) => i);
    if (k >= featureCount) {
      return features;
    }

    for (let i = featureCount - 1; i > 0; i -= 1) {
      const j = Math.floor(this.random() * (i + 1));
      const tmp = features[i];
      features[i] = features[j];
      features[j] = tmp;
    }

    return features.slice(0, k);
  }

  private findBestThreshold(indices: number[], featureIndex: number): SplitCandidate | null {
    const X = this.XTrain!;
    const y = this.yTrain!;
    const sampleCount = indices.length;

    const sortedIndices = [...indices].sort(
      (a, b) => X[a][featureIndex] - X[b][featureIndex],
    );

    let totalPositive = 0;
    for (let i = 0; i < sortedIndices.length; i += 1) {
      totalPositive += y[sortedIndices[i]] === 1 ? 1 : 0;
    }

    let leftCount = 0;
    let leftPositive = 0;
    let bestImpurity = Number.POSITIVE_INFINITY;
    let bestThreshold = 0;

    for (let i = 1; i < sortedIndices.length; i += 1) {
      const previousIndex = sortedIndices[i - 1];
      leftCount += 1;
      leftPositive += y[previousIndex] === 1 ? 1 : 0;
      const rightCount = sampleCount - leftCount;

      if (leftCount < this.minSamplesLeaf || rightCount < this.minSamplesLeaf) {
        continue;
      }

      const leftValue = X[previousIndex][featureIndex];
      const rightValue = X[sortedIndices[i]][featureIndex];
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
    for (let i = 0; i < indices.length; i += 1) {
      const sampleIndex = indices[i];
      if (X[sampleIndex][featureIndex] <= bestThreshold) {
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
}
