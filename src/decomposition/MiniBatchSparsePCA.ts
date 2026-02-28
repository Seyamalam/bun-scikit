import type { Matrix, Vector } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";
import { SparsePCA, type SparsePCAOptions } from "./SparsePCA";

export interface MiniBatchSparsePCAOptions extends SparsePCAOptions {
  batchSize?: number;
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

function computeFeatureMeans(X: Matrix): Vector {
  const means = new Array<number>(X[0].length).fill(0);
  for (let i = 0; i < X.length; i += 1) {
    for (let j = 0; j < X[i].length; j += 1) {
      means[j] += X[i][j];
    }
  }
  for (let j = 0; j < means.length; j += 1) {
    means[j] /= X.length;
  }
  return means;
}

function normalize(row: number[]): number[] {
  let normSquared = 0;
  for (let i = 0; i < row.length; i += 1) {
    normSquared += row[i] * row[i];
  }
  const norm = Math.sqrt(normSquared);
  if (norm <= 1e-12) {
    return row.slice();
  }
  return row.map((value) => value / norm);
}

function softThreshold(value: number, alpha: number): number {
  if (value > alpha) {
    return value - alpha;
  }
  if (value < -alpha) {
    return value + alpha;
  }
  return 0;
}

export class MiniBatchSparsePCA extends SparsePCA {
  private batchSize: number;
  private readonly options: MiniBatchSparsePCAOptions;

  constructor(options: MiniBatchSparsePCAOptions = {}) {
    super(options);
    this.options = { ...options };
    this.batchSize = options.batchSize ?? 64;
    if (!Number.isInteger(this.batchSize) || this.batchSize < 1) {
      throw new Error(`batchSize must be an integer >= 1. Got ${this.batchSize}.`);
    }
  }

  override fit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    const nSamples = X.length;
    const nFeatures = X[0].length;
    const batchSize = Math.min(this.batchSize, nSamples);
    const random = this.options.randomState === undefined ? Math.random : mulberry32(this.options.randomState);
    const epochs = Math.max(1, Math.floor((this.options.maxIter ?? 1000) / Math.max(1, Math.ceil(nSamples / batchSize))));

    let runningComponents: Matrix | null = null;
    let updateCount = 0;
    for (let epoch = 0; epoch < epochs; epoch += 1) {
      for (let start = 0; start < nSamples; start += batchSize) {
        const batch: Matrix = new Array(batchSize);
        for (let i = 0; i < batchSize; i += 1) {
          batch[i] = X[Math.floor(random() * nSamples)];
        }

        const batchModel = new SparsePCA({
          nComponents: this.options.nComponents,
          alpha: this.options.alpha,
          maxIter: Math.max(64, Math.floor((this.options.maxIter ?? 1000) / 8)),
          tolerance: this.options.tolerance,
          randomState: this.options.randomState === undefined
            ? undefined
            : this.options.randomState + updateCount + 1,
        }).fit(batch);

        const batchComponents = batchModel.components_!;
        if (!runningComponents) {
          runningComponents = batchComponents.map((row) => row.slice());
        } else {
          const eta = 1 / Math.sqrt(updateCount + 1);
          const alpha = this.options.alpha ?? 1;
          for (let c = 0; c < runningComponents.length; c += 1) {
            const sign = dotSign(runningComponents[c], batchComponents[c]) >= 0 ? 1 : -1;
            for (let j = 0; j < nFeatures; j += 1) {
              const updated =
                (1 - eta) * runningComponents[c][j] + eta * sign * batchComponents[c][j];
              runningComponents[c][j] = softThreshold(updated, alpha * eta * 0.1);
            }
            runningComponents[c] = normalize(runningComponents[c]);
          }
        }
        updateCount += 1;
      }
    }

    if (!runningComponents) {
      throw new Error("MiniBatchSparsePCA failed to build components.");
    }

    this.components_ = runningComponents;
    this.mean_ = computeFeatureMeans(X);
    this.nFeaturesIn_ = nFeatures;
    this.nComponents_ = runningComponents.length;
    (this as unknown as { fitted: boolean }).fitted = true;
    return this;
  }
}

function dotSign(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    sum += a[i] * b[i];
  }
  return sum;
}
