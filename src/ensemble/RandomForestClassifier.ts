import type { ClassificationModel, Matrix, Vector } from "../types";
import { accuracyScore } from "../metrics/classification";
import { DecisionTreeClassifier, type MaxFeaturesOption } from "../tree/DecisionTreeClassifier";
import { assertFiniteVector, validateClassificationInputs } from "../utils/validation";

export interface RandomForestClassifierOptions {
  nEstimators?: number;
  maxDepth?: number;
  minSamplesSplit?: number;
  minSamplesLeaf?: number;
  maxFeatures?: MaxFeaturesOption;
  bootstrap?: boolean;
  randomState?: number;
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

export class RandomForestClassifier implements ClassificationModel {
  classes_: Vector = [0, 1];
  private readonly nEstimators: number;
  private readonly maxDepth?: number;
  private readonly minSamplesSplit?: number;
  private readonly minSamplesLeaf?: number;
  private readonly maxFeatures: MaxFeaturesOption;
  private readonly bootstrap: boolean;
  private readonly randomState?: number;
  private trees: DecisionTreeClassifier[] = [];

  constructor(options: RandomForestClassifierOptions = {}) {
    this.nEstimators = options.nEstimators ?? 50;
    this.maxDepth = options.maxDepth ?? 12;
    this.minSamplesSplit = options.minSamplesSplit ?? 2;
    this.minSamplesLeaf = options.minSamplesLeaf ?? 1;
    this.maxFeatures = options.maxFeatures ?? "sqrt";
    this.bootstrap = options.bootstrap ?? true;
    this.randomState = options.randomState;

    if (!Number.isInteger(this.nEstimators) || this.nEstimators < 1) {
      throw new Error(`nEstimators must be a positive integer. Got ${this.nEstimators}.`);
    }
  }

  fit(X: Matrix, y: Vector): this {
    validateClassificationInputs(X, y);

    const sampleCount = X.length;
    const featureCount = X[0].length;
    const random = this.randomState === undefined ? Math.random : mulberry32(this.randomState);
    const flattenedX = this.flattenTrainingMatrix(X, sampleCount, featureCount);
    const yBinary = this.buildBinaryTargets(y);
    const sampleIndices = new Uint32Array(sampleCount);
    this.trees = new Array(this.nEstimators);

    for (let estimatorIndex = 0; estimatorIndex < this.nEstimators; estimatorIndex += 1) {
      if (this.bootstrap) {
        for (let i = 0; i < sampleCount; i += 1) {
          sampleIndices[i] = Math.floor(random() * sampleCount);
        }
      } else {
        for (let i = 0; i < sampleCount; i += 1) {
          sampleIndices[i] = i;
        }
      }

      const tree = new DecisionTreeClassifier({
        maxDepth: this.maxDepth,
        minSamplesSplit: this.minSamplesSplit,
        minSamplesLeaf: this.minSamplesLeaf,
        maxFeatures: this.maxFeatures,
        randomState:
          this.randomState === undefined ? undefined : this.randomState + estimatorIndex + 1,
      });
      tree.fit(X, y, sampleIndices, true, flattenedX, yBinary);
      this.trees[estimatorIndex] = tree;
    }

    return this;
  }

  predict(X: Matrix): Vector {
    if (this.trees.length === 0) {
      throw new Error("RandomForestClassifier has not been fitted.");
    }

    const sampleCount = X.length;
    const voteCounts = new Uint16Array(sampleCount);

    for (let treeIndex = 0; treeIndex < this.trees.length; treeIndex += 1) {
      const treePrediction = this.trees[treeIndex].predict(X);
      for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex += 1) {
        if (treePrediction[sampleIndex] === 1) {
          voteCounts[sampleIndex] += 1;
        }
      }
    }

    const predictions = new Array<number>(sampleCount);
    const voteThreshold = this.trees.length;
    for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex += 1) {
      predictions[sampleIndex] = voteCounts[sampleIndex] * 2 >= voteThreshold ? 1 : 0;
    }

    return predictions;
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return accuracyScore(y, this.predict(X));
  }

  dispose(): void {
    for (let i = 0; i < this.trees.length; i += 1) {
      this.trees[i].dispose();
    }
    this.trees = [];
  }

  private flattenTrainingMatrix(
    X: Matrix,
    sampleCount: number,
    featureCount: number,
  ): Float64Array {
    const flattened = new Float64Array(sampleCount * featureCount);
    for (let i = 0; i < sampleCount; i += 1) {
      const row = X[i];
      const rowOffset = i * featureCount;
      for (let j = 0; j < featureCount; j += 1) {
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
