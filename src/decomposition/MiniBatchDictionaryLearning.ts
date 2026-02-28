import type { Matrix, Vector } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";
import {
  DictionaryLearning,
  type DictionaryLearningOptions,
} from "./DictionaryLearning";

export interface MiniBatchDictionaryLearningOptions extends DictionaryLearningOptions {
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

function centerMatrix(X: Matrix, mean: Vector): Matrix {
  const out: Matrix = new Array(X.length);
  for (let i = 0; i < X.length; i += 1) {
    const row = new Array<number>(X[0].length);
    for (let j = 0; j < X[0].length; j += 1) {
      row[j] = X[i][j] - mean[j];
    }
    out[i] = row;
  }
  return out;
}

function matrixMultiply(A: Matrix, B: Matrix): Matrix {
  const out: Matrix = Array.from({ length: A.length }, () => new Array<number>(B[0].length).fill(0));
  for (let i = 0; i < A.length; i += 1) {
    for (let k = 0; k < A[0].length; k += 1) {
      const value = A[i][k];
      for (let j = 0; j < B[0].length; j += 1) {
        out[i][j] += value * B[k][j];
      }
    }
  }
  return out;
}

function meanSquaredResidual(X: Matrix, code: Matrix, dictionary: Matrix): number {
  const reconstruction = matrixMultiply(code, dictionary);
  let error = 0;
  for (let i = 0; i < X.length; i += 1) {
    for (let j = 0; j < X[0].length; j += 1) {
      const diff = X[i][j] - reconstruction[i][j];
      error += diff * diff;
    }
  }
  return error / (X.length * X[0].length);
}

function normalizeRowInPlace(row: number[]): void {
  let normSquared = 0;
  for (let i = 0; i < row.length; i += 1) {
    normSquared += row[i] * row[i];
  }
  const norm = Math.sqrt(normSquared);
  if (norm <= 1e-12) {
    return;
  }
  for (let i = 0; i < row.length; i += 1) {
    row[i] /= norm;
  }
}

export class MiniBatchDictionaryLearning extends DictionaryLearning {
  private batchSize: number;

  constructor(options: MiniBatchDictionaryLearningOptions = {}) {
    super(options);
    this.batchSize = options.batchSize ?? 64;
    if (!Number.isInteger(this.batchSize) || this.batchSize < 1) {
      throw new Error(`batchSize must be an integer >= 1. Got ${this.batchSize}.`);
    }
  }

  override fit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    const nFeatures = X[0].length;
    const nComponents = this.resolveNComponents(nFeatures);
    const mean = computeFeatureMeans(X);
    const centered = centerMatrix(X, mean);

    const rng = this.randomState === undefined ? Math.random : mulberry32(this.randomState);
    const dictionary = this.initializeDictionary(centered, nComponents, rng);
    const batchSize = Math.min(this.batchSize, X.length);

    let previousError = Number.POSITIVE_INFINITY;
    let bestError = previousError;

    for (let iter = 0; iter < this.maxIter; iter += 1) {
      const batch: Matrix = new Array(batchSize);
      for (let i = 0; i < batchSize; i += 1) {
        batch[i] = centered[Math.floor(rng() * centered.length)];
      }

      const code = this.encodeSparse(batch, dictionary, this.alpha);
      const reconstruction = matrixMultiply(code, dictionary);
      const residual: Matrix = new Array(batchSize);
      for (let i = 0; i < batchSize; i += 1) {
        const row = new Array<number>(nFeatures);
        for (let j = 0; j < nFeatures; j += 1) {
          row[j] = batch[i][j] - reconstruction[i][j];
        }
        residual[i] = row;
      }

      const learningRate = 1 / Math.sqrt(iter + 1);
      for (let component = 0; component < nComponents; component += 1) {
        const gradient = new Array<number>(nFeatures).fill(0);
        for (let i = 0; i < batchSize; i += 1) {
          const coeff = code[i][component];
          for (let j = 0; j < nFeatures; j += 1) {
            gradient[j] += coeff * residual[i][j];
          }
        }
        for (let j = 0; j < nFeatures; j += 1) {
          dictionary[component][j] += (learningRate * gradient[j]) / batchSize;
        }
        normalizeRowInPlace(dictionary[component]);
      }

      if ((iter + 1) % Math.max(5, Math.floor(this.maxIter / 20)) === 0 || iter === this.maxIter - 1) {
        const fullCode = this.encodeSparse(centered, dictionary, this.alpha);
        const mse = meanSquaredResidual(centered, fullCode, dictionary);
        bestError = mse;
        if (Math.abs(previousError - mse) <= this.tolerance) {
          break;
        }
        previousError = mse;
      }
    }

    if (!Number.isFinite(bestError)) {
      const finalCode = this.encodeSparse(centered, dictionary, this.alpha);
      bestError = meanSquaredResidual(centered, finalCode, dictionary);
    }

    this.finalizeFit(dictionary, mean, bestError, nFeatures);
    return this;
  }
}
