import type { Matrix, Vector } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";

export type KernelDensityKernel =
  | "gaussian"
  | "tophat"
  | "epanechnikov"
  | "exponential"
  | "linear"
  | "cosine";

export interface KernelDensityOptions {
  bandwidth?: number;
  kernel?: KernelDensityKernel;
}

function euclideanDistance(a: Vector, b: Vector): number {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

function kernelValue(kernel: KernelDensityKernel, scaledDistance: number): number {
  switch (kernel) {
    case "gaussian":
      return Math.exp(-0.5 * scaledDistance * scaledDistance) / Math.sqrt(2 * Math.PI);
    case "tophat":
      return scaledDistance <= 1 ? 0.5 : 0;
    case "epanechnikov":
      return scaledDistance <= 1 ? 0.75 * (1 - scaledDistance * scaledDistance) : 0;
    case "exponential":
      return 0.5 * Math.exp(-scaledDistance);
    case "linear":
      return scaledDistance <= 1 ? 1 - scaledDistance : 0;
    case "cosine":
      return scaledDistance <= 1 ? (Math.PI / 4) * Math.cos((Math.PI / 2) * scaledDistance) : 0;
    default:
      return 0;
  }
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

function gaussianRandom(random: () => number): number {
  const u1 = Math.max(random(), 1e-12);
  const u2 = random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

export class KernelDensity {
  nFeaturesIn_: number | null = null;

  private bandwidth: number;
  private kernel: KernelDensityKernel;
  private XTrain: Matrix | null = null;

  constructor(options: KernelDensityOptions = {}) {
    this.bandwidth = options.bandwidth ?? 1;
    this.kernel = options.kernel ?? "gaussian";
    if (!Number.isFinite(this.bandwidth) || this.bandwidth <= 0) {
      throw new Error(`bandwidth must be finite and > 0. Got ${this.bandwidth}.`);
    }
  }

  fit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    this.XTrain = X.map((row) => row.slice());
    this.nFeaturesIn_ = X[0].length;
    return this;
  }

  scoreSamples(X: Matrix): Vector {
    this.assertFitted();
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }

    const h = this.bandwidth;
    const n = this.XTrain!.length;
    const d = this.nFeaturesIn_!;
    const scale = 1 / (n * Math.pow(h, d));

    return X.map((sample) => {
      let sum = 0;
      for (let i = 0; i < this.XTrain!.length; i += 1) {
        const distance = euclideanDistance(sample, this.XTrain![i]);
        sum += kernelValue(this.kernel, distance / h);
      }
      return Math.log(Math.max(1e-300, sum * scale));
    });
  }

  score(X: Matrix): number {
    const logDensities = this.scoreSamples(X);
    let sum = 0;
    for (let i = 0; i < logDensities.length; i += 1) {
      sum += logDensities[i];
    }
    return sum;
  }

  sample(nSamples = 1, randomState?: number): Matrix {
    this.assertFitted();
    if (!Number.isInteger(nSamples) || nSamples < 1) {
      throw new Error(`nSamples must be an integer >= 1. Got ${nSamples}.`);
    }
    const random =
      randomState === undefined ? Math.random : mulberry32(randomState);

    const out: Matrix = new Array(nSamples);
    for (let i = 0; i < nSamples; i += 1) {
      const base = this.XTrain![Math.floor(random() * this.XTrain!.length)];
      const row = new Array<number>(this.nFeaturesIn_!);
      for (let j = 0; j < row.length; j += 1) {
        if (this.kernel === "gaussian") {
          row[j] = base[j] + gaussianRandom(random) * this.bandwidth;
        } else {
          row[j] = base[j] + (random() * 2 - 1) * this.bandwidth;
        }
      }
      out[i] = row;
    }
    return out;
  }

  private assertFitted(): void {
    if (!this.XTrain || this.nFeaturesIn_ === null) {
      throw new Error("KernelDensity has not been fitted.");
    }
  }
}
