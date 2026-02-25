import type { ClassificationModel, Matrix, Vector } from "../types";
import { accuracyScore } from "../metrics/classification";
import { DecisionTreeClassifier, type MaxFeaturesOption } from "../tree/DecisionTreeClassifier";
import { assertFiniteVector, validateClassificationInputs } from "../utils/validation";
import { getZigKernels } from "../native/zigKernels";

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

export class RandomForestClassifier implements ClassificationModel {
  classes_: Vector = [0, 1];
  fitBackend_: "zig" | "js" = "js";
  fitBackendLibrary_: string | null = null;
  private readonly nEstimators: number;
  private readonly maxDepth?: number;
  private readonly minSamplesSplit?: number;
  private readonly minSamplesLeaf?: number;
  private readonly maxFeatures: MaxFeaturesOption;
  private readonly bootstrap: boolean;
  private readonly randomState?: number;
  private nativeModelHandle: bigint | null = null;
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
    this.disposeNativeModel();
    validateClassificationInputs(X, y);

    const sampleCount = X.length;
    const featureCount = X[0].length;
    const random = this.randomState === undefined ? Math.random : mulberry32(this.randomState);
    const flattenedX = this.flattenTrainingMatrix(X, sampleCount, featureCount);
    const yBinary = this.buildBinaryTargets(y);
    const sampleIndices = new Uint32Array(sampleCount);
    this.trees = [];
    if (
      isZigTreeBackendEnabled() &&
      this.tryFitNativeForest(flattenedX, yBinary, sampleCount, featureCount)
    ) {
      this.fitBackend_ = "zig";
      return this;
    }
    this.fitBackend_ = "js";
    this.fitBackendLibrary_ = null;
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
    if (this.nativeModelHandle !== null) {
      const kernels = getZigKernels();
      const predict = kernels?.randomForestClassifierModelPredict;
      if (predict) {
        const sampleCount = X.length;
        const featureCount = X[0]?.length ?? 0;
        const flattened = this.flattenTrainingMatrix(X, sampleCount, featureCount);
        const out = new Uint8Array(sampleCount);
        const status = predict(
          this.nativeModelHandle,
          flattened,
          sampleCount,
          featureCount,
          out,
        );
        if (status === 1) {
          return Array.from(out);
        }
      }
    }

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
    this.disposeNativeModel();
    for (let i = 0; i < this.trees.length; i += 1) {
      this.trees[i].dispose();
    }
    this.trees = [];
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

  private tryFitNativeForest(
    flattenedX: Float64Array,
    yBinary: Uint8Array,
    sampleCount: number,
    featureCount: number,
  ): boolean {
    const kernels = getZigKernels();
    const create = kernels?.randomForestClassifierModelCreate;
    const fit = kernels?.randomForestClassifierModelFit;
    const destroy = kernels?.randomForestClassifierModelDestroy;
    if (!create || !fit || !destroy) {
      return false;
    }

    const { mode, value } = this.resolveNativeMaxFeatures(featureCount);
    const useRandomState = this.randomState === undefined ? 0 : 1;
    const randomState = this.randomState ?? 0;
    const handle = create(
      this.nEstimators,
      this.maxDepth ?? 12,
      this.minSamplesSplit ?? 2,
      this.minSamplesLeaf ?? 1,
      mode,
      value,
      this.bootstrap ? 1 : 0,
      randomState >>> 0,
      useRandomState,
      featureCount,
    );
    if (handle === 0n) {
      return false;
    }

    let shouldDestroy = true;
    try {
      const status = fit(handle, flattenedX, yBinary, sampleCount, featureCount);
      if (status !== 1) {
        return false;
      }
      this.nativeModelHandle = handle;
      this.fitBackendLibrary_ = kernels.libraryPath;
      shouldDestroy = false;
      return true;
    } finally {
      if (shouldDestroy) {
        destroy(handle);
      }
    }
  }

  private disposeNativeModel(): void {
    if (this.nativeModelHandle === null) {
      return;
    }
    const kernels = getZigKernels();
    const destroy = kernels?.randomForestClassifierModelDestroy;
    if (destroy) {
      try {
        destroy(this.nativeModelHandle);
      } catch {
        // best effort cleanup
      }
    }
    this.nativeModelHandle = null;
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
