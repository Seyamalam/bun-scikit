import type { Matrix, Vector } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";

export type KernelPCAKernel = "linear" | "rbf" | "poly";

export interface KernelPCAOptions {
  nComponents?: number;
  kernel?: KernelPCAKernel;
  gamma?: number;
  degree?: number;
  coef0?: number;
  tolerance?: number;
  maxIter?: number;
}

function dot(a: Vector, b: Vector): number {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    sum += a[i] * b[i];
  }
  return sum;
}

function squaredDistance(a: Vector, b: Vector): number {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

function identity(size: number): Matrix {
  const out: Matrix = Array.from({ length: size }, () => new Array<number>(size).fill(0));
  for (let i = 0; i < size; i += 1) {
    out[i][i] = 1;
  }
  return out;
}

function cloneMatrix(X: Matrix): Matrix {
  return X.map((row) => row.slice());
}

function jacobiEigenDecomposition(
  symmetric: Matrix,
  tolerance: number,
  maxIter: number,
): { eigenvalues: Vector; eigenvectors: Matrix } {
  const n = symmetric.length;
  const A = cloneMatrix(symmetric);
  const V = identity(n);

  for (let iter = 0; iter < maxIter; iter += 1) {
    let p = 0;
    let q = 1;
    let maxOff = 0;
    for (let i = 0; i < n; i += 1) {
      for (let j = i + 1; j < n; j += 1) {
        const value = Math.abs(A[i][j]);
        if (value > maxOff) {
          maxOff = value;
          p = i;
          q = j;
        }
      }
    }
    if (maxOff <= tolerance) {
      break;
    }

    const app = A[p][p];
    const aqq = A[q][q];
    const apq = A[p][q];
    const phi = 0.5 * Math.atan2(2 * apq, aqq - app);
    const c = Math.cos(phi);
    const s = Math.sin(phi);

    for (let i = 0; i < n; i += 1) {
      if (i === p || i === q) {
        continue;
      }
      const aip = A[i][p];
      const aiq = A[i][q];
      const rip = c * aip - s * aiq;
      const riq = s * aip + c * aiq;
      A[i][p] = rip;
      A[p][i] = rip;
      A[i][q] = riq;
      A[q][i] = riq;
    }

    A[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
    A[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
    A[p][q] = 0;
    A[q][p] = 0;

    for (let i = 0; i < n; i += 1) {
      const vip = V[i][p];
      const viq = V[i][q];
      V[i][p] = c * vip - s * viq;
      V[i][q] = s * vip + c * viq;
    }
  }

  const eigenvalues = new Array<number>(n);
  for (let i = 0; i < n; i += 1) {
    eigenvalues[i] = A[i][i];
  }
  return { eigenvalues, eigenvectors: V };
}

export class KernelPCA {
  alphas_: Matrix | null = null;
  lambdas_: Vector | null = null;
  nFeaturesIn_: number | null = null;

  private readonly nComponents?: number;
  private readonly kernel: KernelPCAKernel;
  private readonly gamma?: number;
  private readonly degree: number;
  private readonly coef0: number;
  private readonly tolerance: number;
  private readonly maxIter: number;
  private XFit_: Matrix | null = null;
  private kFitRowMeans_: Vector | null = null;
  private kFitTotalMean_: number | null = null;
  private isFitted = false;

  constructor(options: KernelPCAOptions = {}) {
    this.nComponents = options.nComponents;
    this.kernel = options.kernel ?? "rbf";
    this.gamma = options.gamma;
    this.degree = options.degree ?? 3;
    this.coef0 = options.coef0 ?? 1;
    this.tolerance = options.tolerance ?? 1e-12;
    this.maxIter = options.maxIter ?? 10_000;

    if (this.nComponents !== undefined && (!Number.isInteger(this.nComponents) || this.nComponents < 1)) {
      throw new Error(`nComponents must be an integer >= 1 when provided. Got ${this.nComponents}.`);
    }
  }

  fit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    const nSamples = X.length;
    const nFeatures = X[0].length;
    const gamma = this.gamma ?? 1 / nFeatures;
    const K = this.kernelMatrix(X, X, gamma);
    const rowMeans = new Array<number>(nSamples).fill(0);
    let totalMean = 0;
    for (let i = 0; i < nSamples; i += 1) {
      let rowSum = 0;
      for (let j = 0; j < nSamples; j += 1) {
        rowSum += K[i][j];
      }
      rowMeans[i] = rowSum / nSamples;
      totalMean += rowMeans[i];
    }
    totalMean /= nSamples;

    const Kc = new Array<Matrix[number]>(nSamples);
    for (let i = 0; i < nSamples; i += 1) {
      const row = new Array<number>(nSamples);
      for (let j = 0; j < nSamples; j += 1) {
        row[j] = K[i][j] - rowMeans[i] - rowMeans[j] + totalMean;
      }
      Kc[i] = row;
    }

    const { eigenvalues, eigenvectors } = jacobiEigenDecomposition(
      Kc,
      this.tolerance,
      this.maxIter,
    );
    const order = Array.from({ length: eigenvalues.length }, (_, i) => i).sort(
      (a, b) => eigenvalues[b] - eigenvalues[a],
    );
    const maxComponents = Math.min(nSamples, this.nComponents ?? nSamples);

    const lambdas: Vector = [];
    const alphas: Matrix = Array.from({ length: nSamples }, () => []);
    for (let c = 0; c < maxComponents; c += 1) {
      const idx = order[c];
      const lambda = Math.max(0, eigenvalues[idx]);
      if (lambda <= 1e-14) {
        continue;
      }
      lambdas.push(lambda);
      const norm = Math.sqrt(lambda);
      for (let i = 0; i < nSamples; i += 1) {
        alphas[i].push(eigenvectors[i][idx] / norm);
      }
    }
    if (lambdas.length === 0) {
      throw new Error("KernelPCA failed to extract components.");
    }

    this.alphas_ = alphas;
    this.lambdas_ = lambdas;
    this.XFit_ = X.map((row) => row.slice());
    this.kFitRowMeans_ = rowMeans;
    this.kFitTotalMean_ = totalMean;
    this.nFeaturesIn_ = nFeatures;
    this.isFitted = true;
    return this;
  }

  transform(X: Matrix): Matrix {
    this.assertFitted();
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }

    const gamma = this.gamma ?? 1 / this.nFeaturesIn_!;
    const K = this.kernelMatrix(X, this.XFit_!, gamma);

    const transformed: Matrix = new Array(X.length);
    for (let i = 0; i < X.length; i += 1) {
      let rowMean = 0;
      for (let j = 0; j < this.XFit_!.length; j += 1) {
        rowMean += K[i][j];
      }
      rowMean /= this.XFit_!.length;

      const centered = new Array<number>(this.XFit_!.length);
      for (let j = 0; j < this.XFit_!.length; j += 1) {
        centered[j] = K[i][j] - rowMean - this.kFitRowMeans_![j] + this.kFitTotalMean_!;
      }

      const projection = new Array<number>(this.lambdas_!.length).fill(0);
      for (let c = 0; c < this.lambdas_!.length; c += 1) {
        let value = 0;
        for (let j = 0; j < centered.length; j += 1) {
          value += centered[j] * this.alphas_![j][c];
        }
        projection[c] = value;
      }
      transformed[i] = projection;
    }

    return transformed;
  }

  fitTransform(X: Matrix): Matrix {
    return this.fit(X).transform(X);
  }

  private kernelMatrix(A: Matrix, B: Matrix, gamma: number): Matrix {
    const out: Matrix = new Array(A.length);
    for (let i = 0; i < A.length; i += 1) {
      const row = new Array<number>(B.length);
      for (let j = 0; j < B.length; j += 1) {
        if (this.kernel === "linear") {
          row[j] = dot(A[i], B[j]);
        } else if (this.kernel === "poly") {
          row[j] = Math.pow(gamma * dot(A[i], B[j]) + this.coef0, this.degree);
        } else {
          row[j] = Math.exp(-gamma * squaredDistance(A[i], B[j]));
        }
      }
      out[i] = row;
    }
    return out;
  }

  private assertFitted(): void {
    if (
      !this.isFitted ||
      !this.alphas_ ||
      !this.lambdas_ ||
      !this.XFit_ ||
      !this.kFitRowMeans_ ||
      this.kFitTotalMean_ === null ||
      this.nFeaturesIn_ === null
    ) {
      throw new Error("KernelPCA has not been fitted.");
    }
  }
}
