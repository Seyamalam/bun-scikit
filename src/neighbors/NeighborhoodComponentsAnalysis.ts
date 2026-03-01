import type { Matrix, Vector } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  validateClassificationInputs,
} from "../utils/validation";
import { uniqueSortedLabels } from "../utils/classification";
import { multiplyMatrices, transpose } from "../utils/linalg";

export interface NeighborhoodComponentsAnalysisOptions {
  nComponents?: number | null;
  maxIter?: number;
  tolerance?: number;
  randomState?: number;
}

function normalizeVector(values: Vector): void {
  let norm = 0;
  for (let i = 0; i < values.length; i += 1) {
    norm += values[i] * values[i];
  }
  norm = Math.sqrt(norm);
  if (norm <= 1e-12) {
    const uniform = 1 / Math.sqrt(values.length);
    for (let i = 0; i < values.length; i += 1) {
      values[i] = uniform;
    }
    return;
  }
  for (let i = 0; i < values.length; i += 1) {
    values[i] /= norm;
  }
}

function symmetricDeflation(A: Matrix, vector: Vector, eigenvalue: number): void {
  for (let i = 0; i < A.length; i += 1) {
    for (let j = 0; j < A[i].length; j += 1) {
      A[i][j] -= eigenvalue * vector[i] * vector[j];
    }
  }
}

function multiplyMatrixVector(A: Matrix, x: Vector): Vector {
  const out = new Array<number>(A.length).fill(0);
  for (let i = 0; i < A.length; i += 1) {
    let sum = 0;
    for (let j = 0; j < A[i].length; j += 1) {
      sum += A[i][j] * x[j];
    }
    out[i] = sum;
  }
  return out;
}

function dot(a: Vector, b: Vector): number {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    sum += a[i] * b[i];
  }
  return sum;
}

function seededRandom(seed: number): () => number {
  let state = (seed >>> 0) + 0x9e3779b9;
  return () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function topEigenvectors(
  matrix: Matrix,
  nComponents: number,
  maxIter: number,
  tolerance: number,
  randomState: number,
): Matrix {
  const A = matrix.map((row) => row.slice());
  const random = seededRandom(randomState);
  const vectors: Matrix = new Array(nComponents);

  for (let component = 0; component < nComponents; component += 1) {
    const vector = new Array<number>(A.length).fill(0).map(() => random() * 2 - 1);
    normalizeVector(vector);

    let previous = vector.slice();
    for (let iter = 0; iter < maxIter; iter += 1) {
      const next = multiplyMatrixVector(A, previous);
      normalizeVector(next);
      let delta = 0;
      for (let i = 0; i < next.length; i += 1) {
        delta += Math.abs(next[i] - previous[i]);
      }
      previous = next;
      if (delta < tolerance) {
        break;
      }
    }

    const Av = multiplyMatrixVector(A, previous);
    const eigenvalue = dot(previous, Av);
    vectors[component] = previous.slice();
    symmetricDeflation(A, previous, eigenvalue);
  }

  return vectors;
}

export class NeighborhoodComponentsAnalysis {
  components_: Matrix | null = null;
  classes_: Vector | null = null;
  nFeaturesIn_: number | null = null;
  nIter_ = 0;

  private nComponents: number | null;
  private maxIter: number;
  private tolerance: number;
  private randomState: number;
  private fitted = false;

  constructor(options: NeighborhoodComponentsAnalysisOptions = {}) {
    this.nComponents = options.nComponents ?? null;
    this.maxIter = options.maxIter ?? 200;
    this.tolerance = options.tolerance ?? 1e-5;
    this.randomState = options.randomState ?? 0;

    if (this.nComponents !== null && (!Number.isInteger(this.nComponents) || this.nComponents < 1)) {
      throw new Error(`nComponents must be null or an integer >= 1. Got ${this.nComponents}.`);
    }
    if (!Number.isInteger(this.maxIter) || this.maxIter < 1) {
      throw new Error(`maxIter must be an integer >= 1. Got ${this.maxIter}.`);
    }
    if (!Number.isFinite(this.tolerance) || this.tolerance <= 0) {
      throw new Error(`tolerance must be finite and > 0. Got ${this.tolerance}.`);
    }
  }

  fit(X: Matrix, y: Vector): this {
    validateClassificationInputs(X, y);
    const classes = uniqueSortedLabels(y);
    if (classes.length < 2) {
      throw new Error("NeighborhoodComponentsAnalysis requires at least two classes.");
    }

    const nSamples = X.length;
    const nFeatures = X[0].length;
    const globalMean = new Array<number>(nFeatures).fill(0);
    for (let i = 0; i < nSamples; i += 1) {
      for (let j = 0; j < nFeatures; j += 1) {
        globalMean[j] += X[i][j];
      }
    }
    for (let j = 0; j < nFeatures; j += 1) {
      globalMean[j] /= nSamples;
    }

    const scatter: Matrix = Array.from({ length: nFeatures }, () => new Array<number>(nFeatures).fill(0));
    for (let c = 0; c < classes.length; c += 1) {
      const classSamples = X.filter((_, idx) => y[idx] === classes[c]);
      const count = classSamples.length;
      const classMean = new Array<number>(nFeatures).fill(0);
      for (let i = 0; i < classSamples.length; i += 1) {
        for (let j = 0; j < nFeatures; j += 1) {
          classMean[j] += classSamples[i][j];
        }
      }
      for (let j = 0; j < nFeatures; j += 1) {
        classMean[j] /= Math.max(1, count);
      }
      const diff = new Array<number>(nFeatures);
      for (let j = 0; j < nFeatures; j += 1) {
        diff[j] = classMean[j] - globalMean[j];
      }
      for (let i = 0; i < nFeatures; i += 1) {
        for (let j = 0; j < nFeatures; j += 1) {
          scatter[i][j] += count * diff[i] * diff[j];
        }
      }
    }

    const nComponents = this.nComponents ?? Math.min(nFeatures, Math.max(1, classes.length - 1));
    const components = topEigenvectors(scatter, nComponents, this.maxIter, this.tolerance, this.randomState);

    this.components_ = components;
    this.classes_ = classes;
    this.nFeaturesIn_ = nFeatures;
    this.nIter_ = this.maxIter;
    this.fitted = true;
    return this;
  }

  transform(X: Matrix): Matrix {
    this.assertFitted();
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }
    return multiplyMatrices(X, transpose(this.components_!));
  }

  fitTransform(X: Matrix, y: Vector): Matrix {
    return this.fit(X, y).transform(X);
  }

  getMahalanobisMatrix(): Matrix {
    this.assertFitted();
    const components = this.components_!;
    return multiplyMatrices(transpose(components), components);
  }

  score(X: Matrix, y: Vector): number {
    this.assertFitted();
    assertFiniteVector(y);
    const embedded = this.transform(X);
    if (embedded.length !== y.length) {
      throw new Error("X and y length mismatch.");
    }
    // Proxy score: average class-conditional separation along transformed dimensions.
    const classes = this.classes_!;
    let separation = 0;
    let pairCount = 0;
    for (let i = 0; i < classes.length; i += 1) {
      const classI = embedded.filter((_, idx) => y[idx] === classes[i]);
      if (classI.length === 0) {
        continue;
      }
      const meanI = new Array<number>(embedded[0].length).fill(0);
      for (let r = 0; r < classI.length; r += 1) {
        for (let c = 0; c < meanI.length; c += 1) {
          meanI[c] += classI[r][c];
        }
      }
      for (let c = 0; c < meanI.length; c += 1) {
        meanI[c] /= classI.length;
      }

      for (let j = i + 1; j < classes.length; j += 1) {
        const classJ = embedded.filter((_, idx) => y[idx] === classes[j]);
        if (classJ.length === 0) {
          continue;
        }
        const meanJ = new Array<number>(embedded[0].length).fill(0);
        for (let r = 0; r < classJ.length; r += 1) {
          for (let c = 0; c < meanJ.length; c += 1) {
            meanJ[c] += classJ[r][c];
          }
        }
        for (let c = 0; c < meanJ.length; c += 1) {
          meanJ[c] /= classJ.length;
        }
        let dist = 0;
        for (let c = 0; c < meanI.length; c += 1) {
          const d = meanI[c] - meanJ[c];
          dist += d * d;
        }
        separation += Math.sqrt(dist);
        pairCount += 1;
      }
    }
    return pairCount === 0 ? 0 : separation / pairCount;
  }

  private assertFitted(): void {
    if (!this.fitted || !this.components_ || !this.classes_ || this.nFeaturesIn_ === null) {
      throw new Error("NeighborhoodComponentsAnalysis has not been fitted.");
    }
  }
}

