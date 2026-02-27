import type { Matrix, Vector } from "../types";
import { dot } from "../utils/linalg";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";
import { PCA } from "./PCA";

export interface FastICAOptions {
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

function normalize(vector: Vector): Vector {
  let norm = 0;
  for (let i = 0; i < vector.length; i += 1) {
    norm += vector[i] * vector[i];
  }
  norm = Math.sqrt(norm);
  if (norm === 0) {
    return vector.slice();
  }
  return vector.map((value) => value / norm);
}

function orthonormalizeRows(matrix: Matrix): Matrix {
  const out: Matrix = new Array(matrix.length);
  for (let i = 0; i < matrix.length; i += 1) {
    let row = matrix[i].slice();
    for (let j = 0; j < i; j += 1) {
      const projection = dot(row, out[j]);
      for (let k = 0; k < row.length; k += 1) {
        row[k] -= projection * out[j][k];
      }
    }
    row = normalize(row);
    out[i] = row;
  }
  return out;
}

function transpose(X: Matrix): Matrix {
  const rows = X.length;
  const cols = X[0].length;
  const out: Matrix = Array.from({ length: cols }, () => new Array(rows).fill(0));
  for (let i = 0; i < rows; i += 1) {
    for (let j = 0; j < cols; j += 1) {
      out[j][i] = X[i][j];
    }
  }
  return out;
}

export class FastICA {
  components_: Matrix | null = null;
  mixing_: Matrix | null = null;
  mean_: Vector | null = null;
  nIter_: number | null = null;
  nFeaturesIn_: number | null = null;
  nComponents_: number | null = null;

  private readonly nComponents?: number;
  private readonly maxIter: number;
  private readonly tolerance: number;
  private readonly randomState?: number;
  private pcaModel: PCA | null = null;
  private unmixing: Matrix | null = null;
  private isFitted = false;

  constructor(options: FastICAOptions = {}) {
    this.nComponents = options.nComponents;
    this.maxIter = options.maxIter ?? 400;
    this.tolerance = options.tolerance ?? 1e-5;
    this.randomState = options.randomState;

    if (this.nComponents !== undefined && (!Number.isInteger(this.nComponents) || this.nComponents < 1)) {
      throw new Error(`nComponents must be an integer >= 1 when provided. Got ${this.nComponents}.`);
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

    const nSamples = X.length;
    const nFeatures = X[0].length;
    const maxComponents = Math.min(nSamples, nFeatures);
    const componentCount = this.nComponents ?? maxComponents;
    if (componentCount > maxComponents) {
      throw new Error(
        `nComponents (${componentCount}) cannot exceed min(nSamples, nFeatures) (${maxComponents}).`,
      );
    }

    const pca = new PCA({ nComponents: componentCount, whiten: true });
    const whitened = pca.fitTransform(X);
    const random =
      this.randomState === undefined
        ? Math.random
        : (() => {
            const rng = new Mulberry32(this.randomState!);
            return () => rng.next();
          })();

    let W: Matrix = new Array(componentCount);
    for (let i = 0; i < componentCount; i += 1) {
      const row = new Array<number>(componentCount);
      for (let j = 0; j < componentCount; j += 1) {
        row[j] = random() * 2 - 1;
      }
      W[i] = row;
    }
    W = orthonormalizeRows(W);

    let convergedIter = this.maxIter;
    for (let iter = 0; iter < this.maxIter; iter += 1) {
      const updated: Matrix = new Array(componentCount);
      for (let componentIndex = 0; componentIndex < componentCount; componentIndex += 1) {
        const w = W[componentIndex];
        const accum = new Array<number>(componentCount).fill(0);
        let gPrimeMean = 0;

        for (let sampleIndex = 0; sampleIndex < nSamples; sampleIndex += 1) {
          const z = whitened[sampleIndex];
          const projection = dot(w, z);
          const g = Math.tanh(projection);
          const gPrime = 1 - g * g;
          gPrimeMean += gPrime;
          for (let k = 0; k < componentCount; k += 1) {
            accum[k] += z[k] * g;
          }
        }

        gPrimeMean /= nSamples;
        const newRow = new Array<number>(componentCount);
        for (let k = 0; k < componentCount; k += 1) {
          newRow[k] = accum[k] / nSamples - gPrimeMean * w[k];
        }
        updated[componentIndex] = newRow;
      }

      const WNew = orthonormalizeRows(updated);
      let maxChange = 0;
      for (let i = 0; i < componentCount; i += 1) {
        const alignment = Math.abs(dot(WNew[i], W[i]));
        const change = Math.abs(alignment - 1);
        if (change > maxChange) {
          maxChange = change;
        }
      }
      W = WNew;
      if (maxChange < this.tolerance) {
        convergedIter = iter + 1;
        break;
      }
    }

    this.unmixing = W;
    this.components_ = W.map((row) => row.slice());
    this.mixing_ = transpose(W);
    this.mean_ = pca.mean_ ? pca.mean_.slice() : null;
    this.nIter_ = convergedIter;
    this.nFeaturesIn_ = nFeatures;
    this.nComponents_ = componentCount;
    this.pcaModel = pca;
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

    const whitened = this.pcaModel!.transform(X);
    const sources: Matrix = new Array(whitened.length);
    for (let i = 0; i < whitened.length; i += 1) {
      const row = new Array<number>(this.unmixing!.length);
      for (let j = 0; j < this.unmixing!.length; j += 1) {
        row[j] = dot(whitened[i], this.unmixing![j]);
      }
      sources[i] = row;
    }
    return sources;
  }

  fitTransform(X: Matrix): Matrix {
    return this.fit(X).transform(X);
  }

  inverseTransform(X: Matrix): Matrix {
    this.assertFitted();
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    if (X[0].length !== this.unmixing!.length) {
      throw new Error(
        `Component size mismatch. Expected ${this.unmixing!.length}, got ${X[0].length}.`,
      );
    }

    const whitened: Matrix = new Array(X.length);
    for (let i = 0; i < X.length; i += 1) {
      const row = new Array<number>(this.unmixing!.length).fill(0);
      for (let sourceIndex = 0; sourceIndex < this.unmixing!.length; sourceIndex += 1) {
        const value = X[i][sourceIndex];
        for (let componentIndex = 0; componentIndex < this.unmixing!.length; componentIndex += 1) {
          row[componentIndex] += value * this.unmixing![sourceIndex][componentIndex];
        }
      }
      whitened[i] = row;
    }
    return this.pcaModel!.inverseTransform(whitened);
  }

  private assertFitted(): void {
    if (
      !this.isFitted ||
      !this.pcaModel ||
      !this.unmixing ||
      !this.components_ ||
      !this.mixing_ ||
      this.nFeaturesIn_ === null
    ) {
      throw new Error("FastICA has not been fitted.");
    }
  }
}
