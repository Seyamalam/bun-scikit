import type { Matrix, Vector } from "../types";
import { dot, mean } from "../utils/linalg";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";

export interface TruncatedSVDOptions {
  nComponents?: number;
  nIter?: number;
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

function cloneMatrix(X: Matrix): Matrix {
  return X.map((row) => row.slice());
}

function l2Norm(values: Vector): number {
  let sum = 0;
  for (let i = 0; i < values.length; i += 1) {
    sum += values[i] * values[i];
  }
  return Math.sqrt(sum);
}

function normalize(values: Vector): Vector {
  const norm = l2Norm(values);
  if (norm === 0) {
    return values.slice();
  }
  return values.map((value) => value / norm);
}

function multiplyMatrixVector(matrix: Matrix, vector: Vector): Vector {
  const out = new Array<number>(matrix.length).fill(0);
  for (let i = 0; i < matrix.length; i += 1) {
    let sum = 0;
    for (let j = 0; j < matrix[i].length; j += 1) {
      sum += matrix[i][j] * vector[j];
    }
    out[i] = sum;
  }
  return out;
}

function transformWithComponents(X: Matrix, components: Matrix): Matrix {
  const out: Matrix = new Array(X.length);
  for (let i = 0; i < X.length; i += 1) {
    const row = new Array<number>(components.length);
    for (let j = 0; j < components.length; j += 1) {
      row[j] = dot(X[i], components[j]);
    }
    out[i] = row;
  }
  return out;
}

function featureTotalVariance(X: Matrix): number {
  const nSamples = X.length;
  const nFeatures = X[0].length;
  let total = 0;
  for (let j = 0; j < nFeatures; j += 1) {
    let columnMean = 0;
    for (let i = 0; i < nSamples; i += 1) {
      columnMean += X[i][j];
    }
    columnMean /= nSamples;

    let variance = 0;
    for (let i = 0; i < nSamples; i += 1) {
      const centered = X[i][j] - columnMean;
      variance += centered * centered;
    }
    variance /= nSamples > 1 ? nSamples - 1 : 1;
    total += variance;
  }
  return total;
}

function columnVariance(X: Matrix, column: number): number {
  const n = X.length;
  const values = X.map((row) => row[column]);
  const columnMean = mean(values);
  let sum = 0;
  for (let i = 0; i < n; i += 1) {
    const centered = values[i] - columnMean;
    sum += centered * centered;
  }
  return sum / (n > 1 ? n - 1 : 1);
}

export class TruncatedSVD {
  components_: Matrix | null = null;
  explainedVariance_: Vector | null = null;
  explainedVarianceRatio_: Vector | null = null;
  singularValues_: Vector | null = null;
  nFeaturesIn_: number | null = null;
  nComponents_: number | null = null;

  private readonly nComponents: number;
  private readonly nIter: number;
  private readonly tolerance: number;
  private readonly randomState?: number;
  private isFitted = false;

  constructor(options: TruncatedSVDOptions = {}) {
    this.nComponents = options.nComponents ?? 2;
    this.nIter = options.nIter ?? 10;
    this.tolerance = options.tolerance ?? 1e-7;
    this.randomState = options.randomState;

    if (!Number.isInteger(this.nComponents) || this.nComponents < 1) {
      throw new Error(`nComponents must be an integer >= 1. Got ${this.nComponents}.`);
    }
    if (!Number.isInteger(this.nIter) || this.nIter < 1) {
      throw new Error(`nIter must be an integer >= 1. Got ${this.nIter}.`);
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
    if (this.nComponents > maxComponents) {
      throw new Error(
        `nComponents (${this.nComponents}) cannot exceed min(nSamples, nFeatures) (${maxComponents}).`,
      );
    }

    const gram: Matrix = Array.from({ length: nFeatures }, () => new Array(nFeatures).fill(0));
    for (let i = 0; i < nSamples; i += 1) {
      const row = X[i];
      for (let a = 0; a < nFeatures; a += 1) {
        const xa = row[a];
        for (let b = a; b < nFeatures; b += 1) {
          gram[a][b] += xa * row[b];
        }
      }
    }
    for (let a = 0; a < nFeatures; a += 1) {
      for (let b = 0; b < a; b += 1) {
        gram[a][b] = gram[b][a];
      }
    }

    const working = cloneMatrix(gram);
    const components: Matrix = [];
    const singularValues: Vector = [];
    const random =
      this.randomState === undefined
        ? Math.random
        : (() => {
            const rng = new Mulberry32(this.randomState!);
            return () => rng.next();
          })();

    for (let componentIndex = 0; componentIndex < this.nComponents; componentIndex += 1) {
      let vector = new Array<number>(nFeatures);
      for (let j = 0; j < nFeatures; j += 1) {
        vector[j] = random() * 2 - 1;
      }
      vector = normalize(vector);

      for (let iter = 0; iter < this.nIter; iter += 1) {
        const multiplied = multiplyMatrixVector(working, vector);
        const multipliedNorm = l2Norm(multiplied);
        if (multipliedNorm <= 1e-14) {
          break;
        }
        const nextVector = multiplied.map((value) => value / multipliedNorm);
        let maxDiff = 0;
        for (let j = 0; j < nFeatures; j += 1) {
          const diff = Math.abs(nextVector[j] - vector[j]);
          if (diff > maxDiff) {
            maxDiff = diff;
          }
        }
        vector = nextVector;
        if (maxDiff < this.tolerance) {
          break;
        }
      }

      const projected = multiplyMatrixVector(working, vector);
      const eigenvalue = Math.max(0, dot(vector, projected));
      if (eigenvalue <= 1e-14) {
        break;
      }

      components.push(vector);
      singularValues.push(Math.sqrt(eigenvalue));

      for (let a = 0; a < nFeatures; a += 1) {
        const va = vector[a];
        for (let b = 0; b < nFeatures; b += 1) {
          working[a][b] -= eigenvalue * va * vector[b];
        }
      }
    }

    if (components.length === 0) {
      throw new Error("TruncatedSVD could not extract any component from input data.");
    }

    const transformed = transformWithComponents(X, components);
    const explainedVariance = new Array<number>(components.length);
    for (let i = 0; i < components.length; i += 1) {
      explainedVariance[i] = columnVariance(transformed, i);
    }
    const totalVariance = featureTotalVariance(X);
    const explainedVarianceRatio = explainedVariance.map((value) =>
      totalVariance === 0 ? 0 : value / totalVariance,
    );

    this.components_ = components;
    this.singularValues_ = singularValues;
    this.explainedVariance_ = explainedVariance;
    this.explainedVarianceRatio_ = explainedVarianceRatio;
    this.nFeaturesIn_ = nFeatures;
    this.nComponents_ = components.length;
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

    return transformWithComponents(X, this.components_!);
  }

  fitTransform(X: Matrix): Matrix {
    return this.fit(X).transform(X);
  }

  inverseTransform(X: Matrix): Matrix {
    this.assertFitted();
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    if (X[0].length !== this.components_!.length) {
      throw new Error(
        `Component size mismatch. Expected ${this.components_!.length}, got ${X[0].length}.`,
      );
    }

    const out: Matrix = new Array(X.length);
    for (let i = 0; i < X.length; i += 1) {
      const row = new Array<number>(this.nFeaturesIn_!).fill(0);
      for (let componentIndex = 0; componentIndex < this.components_!.length; componentIndex += 1) {
        const value = X[i][componentIndex];
        const component = this.components_![componentIndex];
        for (let featureIndex = 0; featureIndex < this.nFeaturesIn_!; featureIndex += 1) {
          row[featureIndex] += value * component[featureIndex];
        }
      }
      out[i] = row;
    }
    return out;
  }

  private assertFitted(): void {
    if (
      !this.isFitted ||
      !this.components_ ||
      !this.explainedVariance_ ||
      !this.explainedVarianceRatio_ ||
      !this.singularValues_ ||
      this.nFeaturesIn_ === null
    ) {
      throw new Error("TruncatedSVD has not been fitted.");
    }
  }
}
