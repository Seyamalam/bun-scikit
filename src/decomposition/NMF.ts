import type { Matrix } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";

export interface NMFOptions {
  nComponents?: number;
  maxIter?: number;
  tolerance?: number;
  randomState?: number;
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

export class NMF {
  components_: Matrix | null = null;
  reconstructionErr_: number | null = null;
  nIter_: number | null = null;
  nFeaturesIn_: number | null = null;
  nComponents_: number | null = null;

  private readonly nComponents: number;
  private readonly maxIter: number;
  private readonly tolerance: number;
  private readonly randomState?: number;
  private W_: Matrix | null = null;
  private isFitted = false;

  constructor(options: NMFOptions = {}) {
    this.nComponents = options.nComponents ?? 2;
    this.maxIter = options.maxIter ?? 400;
    this.tolerance = options.tolerance ?? 1e-5;
    this.randomState = options.randomState;

    if (!Number.isInteger(this.nComponents) || this.nComponents < 1) {
      throw new Error(`nComponents must be an integer >= 1. Got ${this.nComponents}.`);
    }
    if (!Number.isInteger(this.maxIter) || this.maxIter < 1) {
      throw new Error(`maxIter must be an integer >= 1. Got ${this.maxIter}.`);
    }
    if (!Number.isFinite(this.tolerance) || this.tolerance <= 0) {
      throw new Error(`tolerance must be finite and > 0. Got ${this.tolerance}.`);
    }
  }

  fit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    assertNonNegativeMatrix(X);

    const nSamples = X.length;
    const nFeatures = X[0].length;
    const random =
      this.randomState === undefined
        ? Math.random
        : (() => {
            const rng = new Mulberry32(this.randomState!);
            return () => rng.next();
          })();

    let W = randomPositiveMatrix(nSamples, this.nComponents, random);
    let H = randomPositiveMatrix(this.nComponents, nFeatures, random);
    const eps = 1e-12;

    let previousErr = Number.POSITIVE_INFINITY;
    let convergedAt = this.maxIter;

    for (let iter = 0; iter < this.maxIter; iter += 1) {
      const WT = transpose(W);
      const WTX = matmul(WT, X);
      const WTW = matmul(WT, W);
      const WTWH = matmul(WTW, H);
      for (let i = 0; i < H.length; i += 1) {
        for (let j = 0; j < H[0].length; j += 1) {
          H[i][j] *= WTX[i][j] / (WTWH[i][j] + eps);
        }
      }

      const HT = transpose(H);
      const XHT = matmul(X, HT);
      const WH = matmul(W, H);
      const WHHT = matmul(WH, HT);
      for (let i = 0; i < W.length; i += 1) {
        for (let j = 0; j < W[0].length; j += 1) {
          W[i][j] *= XHT[i][j] / (WHHT[i][j] + eps);
        }
      }

      const err = frobeniusNormDiff(X, matmul(W, H));
      if (Math.abs(previousErr - err) < this.tolerance) {
        convergedAt = iter + 1;
        previousErr = err;
        break;
      }
      previousErr = err;
    }

    this.components_ = H;
    this.W_ = W;
    this.reconstructionErr_ = previousErr;
    this.nIter_ = convergedAt;
    this.nFeaturesIn_ = nFeatures;
    this.nComponents_ = this.nComponents;
    this.isFitted = true;
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

    let W = zeros(X.length, this.nComponents);
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
    if (!this.isFitted || !this.components_ || this.nFeaturesIn_ === null) {
      throw new Error("NMF has not been fitted.");
    }
  }
}
