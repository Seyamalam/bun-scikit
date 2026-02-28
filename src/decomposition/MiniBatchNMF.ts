import type { Matrix } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";

export interface MiniBatchNMFOptions {
  nComponents?: number;
  maxIter?: number;
  tolerance?: number;
  randomState?: number;
  batchSize?: number;
}

class Mulberry32 {
  private state: number;

  constructor(seed: number) {
    this.state = seed >>> 0;
  }

  next(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let t = this.state ^ (this.state >>> 15);
    t = Math.imul(t, this.state | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
}

function zeros(rows: number, cols: number): Matrix {
  return Array.from({ length: rows }, () => new Array<number>(cols).fill(0));
}

function transpose(X: Matrix): Matrix {
  const out = zeros(X[0].length, X.length);
  for (let i = 0; i < X.length; i += 1) {
    for (let j = 0; j < X[0].length; j += 1) {
      out[j][i] = X[i][j];
    }
  }
  return out;
}

function matmul(A: Matrix, B: Matrix): Matrix {
  const out = zeros(A.length, B[0].length);
  for (let i = 0; i < A.length; i += 1) {
    for (let k = 0; k < B.length; k += 1) {
      const aik = A[i][k];
      for (let j = 0; j < B[0].length; j += 1) {
        out[i][j] += aik * B[k][j];
      }
    }
  }
  return out;
}

function frobeniusNormDiff(A: Matrix, B: Matrix): number {
  let sum = 0;
  for (let i = 0; i < A.length; i += 1) {
    for (let j = 0; j < A[0].length; j += 1) {
      const diff = A[i][j] - B[i][j];
      sum += diff * diff;
    }
  }
  return Math.sqrt(sum);
}

function assertNonNegativeMatrix(X: Matrix, label = "X"): void {
  for (let i = 0; i < X.length; i += 1) {
    for (let j = 0; j < X[i].length; j += 1) {
      if (X[i][j] < 0) {
        throw new Error(`${label} must be non-negative. Found ${X[i][j]} at [${i}, ${j}].`);
      }
    }
  }
}

function randomPositiveMatrix(rows: number, cols: number, random: () => number): Matrix {
  const out = zeros(rows, cols);
  for (let i = 0; i < rows; i += 1) {
    for (let j = 0; j < cols; j += 1) {
      out[i][j] = 0.1 + random();
    }
  }
  return out;
}

function sampleIndices(nSamples: number, batchSize: number, random: () => number): number[] {
  const out = new Array<number>(batchSize);
  for (let i = 0; i < batchSize; i += 1) {
    out[i] = Math.floor(random() * nSamples);
  }
  return out;
}

function subsetRows(X: Matrix, indices: number[]): Matrix {
  const out: Matrix = new Array(indices.length);
  for (let i = 0; i < indices.length; i += 1) {
    out[i] = X[indices[i]];
  }
  return out;
}

export class MiniBatchNMF {
  components_: Matrix | null = null;
  reconstructionErr_: number | null = null;
  nIter_: number | null = null;
  nFeaturesIn_: number | null = null;
  nComponents_: number | null = null;

  private nComponents: number;
  private maxIter: number;
  private tolerance: number;
  private randomState?: number;
  private batchSize: number;
  private W_: Matrix | null = null;
  private fitted = false;

  constructor(options: MiniBatchNMFOptions = {}) {
    this.nComponents = options.nComponents ?? 2;
    this.maxIter = options.maxIter ?? 400;
    this.tolerance = options.tolerance ?? 1e-5;
    this.randomState = options.randomState;
    this.batchSize = options.batchSize ?? 64;

    if (!Number.isInteger(this.nComponents) || this.nComponents < 1) {
      throw new Error(`nComponents must be an integer >= 1. Got ${this.nComponents}.`);
    }
    if (!Number.isInteger(this.maxIter) || this.maxIter < 1) {
      throw new Error(`maxIter must be an integer >= 1. Got ${this.maxIter}.`);
    }
    if (!Number.isFinite(this.tolerance) || this.tolerance <= 0) {
      throw new Error(`tolerance must be finite and > 0. Got ${this.tolerance}.`);
    }
    if (!Number.isInteger(this.batchSize) || this.batchSize < 1) {
      throw new Error(`batchSize must be an integer >= 1. Got ${this.batchSize}.`);
    }
  }

  fit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    assertNonNegativeMatrix(X);

    const nSamples = X.length;
    const nFeatures = X[0].length;
    const batchSize = Math.min(this.batchSize, nSamples);
    const random =
      this.randomState === undefined
        ? Math.random
        : (() => {
            const rng = new Mulberry32(this.randomState!);
            return () => rng.next();
          })();

    const eps = 1e-12;
    const W = randomPositiveMatrix(nSamples, this.nComponents, random);
    const H = randomPositiveMatrix(this.nComponents, nFeatures, random);

    let previousErr = Number.POSITIVE_INFINITY;
    let convergedAt = this.maxIter;

    for (let iter = 0; iter < this.maxIter; iter += 1) {
      const batchIdx = sampleIndices(nSamples, batchSize, random);
      const XBatch = subsetRows(X, batchIdx);
      const WBatch = subsetRows(W, batchIdx);

      const WT = transpose(WBatch);
      const WTX = matmul(WT, XBatch);
      const WTW = matmul(WT, WBatch);
      const WTWH = matmul(WTW, H);
      for (let i = 0; i < H.length; i += 1) {
        for (let j = 0; j < H[0].length; j += 1) {
          H[i][j] *= WTX[i][j] / (WTWH[i][j] + eps);
        }
      }

      const HT = transpose(H);
      const XHT = matmul(XBatch, HT);
      const WHBatch = matmul(WBatch, H);
      const WHHT = matmul(WHBatch, HT);
      for (let i = 0; i < WBatch.length; i += 1) {
        for (let j = 0; j < WBatch[0].length; j += 1) {
          WBatch[i][j] *= XHT[i][j] / (WHHT[i][j] + eps);
        }
      }
      for (let i = 0; i < batchIdx.length; i += 1) {
        W[batchIdx[i]] = WBatch[i];
      }

      if ((iter + 1) % Math.max(10, Math.floor(this.maxIter / 20)) === 0 || iter === this.maxIter - 1) {
        const err = frobeniusNormDiff(X, matmul(W, H));
        if (Math.abs(previousErr - err) < this.tolerance) {
          convergedAt = iter + 1;
          previousErr = err;
          break;
        }
        previousErr = err;
      }
    }

    if (!Number.isFinite(previousErr)) {
      previousErr = frobeniusNormDiff(X, matmul(W, H));
    }

    this.components_ = H;
    this.W_ = W;
    this.reconstructionErr_ = previousErr;
    this.nIter_ = convergedAt;
    this.nFeaturesIn_ = nFeatures;
    this.nComponents_ = this.nComponents;
    this.fitted = true;
    return this;
  }

  transform(X: Matrix): Matrix {
    this.assertFitted();
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    assertNonNegativeMatrix(X);

    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }

    const W = zeros(X.length, this.nComponents);
    for (let i = 0; i < W.length; i += 1) {
      for (let j = 0; j < W[0].length; j += 1) {
        W[i][j] = 1;
      }
    }

    const H = this.components_!;
    const HT = transpose(H);
    const HHT = matmul(H, HT);
    const eps = 1e-12;
    for (let iter = 0; iter < Math.min(200, this.maxIter); iter += 1) {
      const XHT = matmul(X, HT);
      const WHHT = matmul(W, HHT);
      for (let i = 0; i < W.length; i += 1) {
        for (let j = 0; j < W[0].length; j += 1) {
          W[i][j] *= XHT[i][j] / (WHHT[i][j] + eps);
        }
      }
    }
    return W;
  }

  fitTransform(X: Matrix): Matrix {
    this.fit(X);
    return this.W_!.map((row) => row.slice());
  }

  inverseTransform(W: Matrix): Matrix {
    this.assertFitted();
    assertNonEmptyMatrix(W, "W");
    assertConsistentRowSize(W, "W");
    assertFiniteMatrix(W, "W");
    assertNonNegativeMatrix(W, "W");
    if (W[0].length !== this.nComponents_) {
      throw new Error(`Component size mismatch. Expected ${this.nComponents_}, got ${W[0].length}.`);
    }
    return matmul(W, this.components_!);
  }

  private assertFitted(): void {
    if (!this.fitted || !this.components_ || this.nFeaturesIn_ === null) {
      throw new Error("MiniBatchNMF has not been fitted.");
    }
  }
}
