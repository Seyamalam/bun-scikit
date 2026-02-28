import type { Matrix, Vector } from "../types";
import { dot } from "../utils/linalg";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";

export interface DictionaryLearningOptions {
  nComponents?: number;
  alpha?: number;
  maxIter?: number;
  tolerance?: number;
  randomState?: number;
  transformAlpha?: number;
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

function softThreshold(value: number, alpha: number): number {
  if (value > alpha) {
    return value - alpha;
  }
  if (value < -alpha) {
    return value + alpha;
  }
  return 0;
}

function normalizeRow(row: Vector): Vector {
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

function gramMatrix(dictionary: Matrix): Matrix {
  const nComponents = dictionary.length;
  const out: Matrix = Array.from({ length: nComponents }, () => new Array<number>(nComponents).fill(0));
  for (let i = 0; i < nComponents; i += 1) {
    out[i][i] = dot(dictionary[i], dictionary[i]);
    for (let j = i + 1; j < nComponents; j += 1) {
      const value = dot(dictionary[i], dictionary[j]);
      out[i][j] = value;
      out[j][i] = value;
    }
  }
  return out;
}

function largestEigenvalueSymmetric(matrix: Matrix, maxIter = 100): number {
  const n = matrix.length;
  let vector = new Array<number>(n).fill(1 / Math.sqrt(n));
  let eigenvalue = 0;

  for (let iter = 0; iter < maxIter; iter += 1) {
    const next = new Array<number>(n).fill(0);
    for (let i = 0; i < n; i += 1) {
      let sum = 0;
      for (let j = 0; j < n; j += 1) {
        sum += matrix[i][j] * vector[j];
      }
      next[i] = sum;
    }

    let normSquared = 0;
    for (let i = 0; i < n; i += 1) {
      normSquared += next[i] * next[i];
    }
    const norm = Math.sqrt(normSquared);
    if (norm <= 1e-20) {
      return 1;
    }

    for (let i = 0; i < n; i += 1) {
      next[i] /= norm;
    }

    let rayleighNumerator = 0;
    let rayleighDenominator = 0;
    for (let i = 0; i < n; i += 1) {
      let mv = 0;
      for (let j = 0; j < n; j += 1) {
        mv += matrix[i][j] * next[j];
      }
      rayleighNumerator += next[i] * mv;
      rayleighDenominator += next[i] * next[i];
    }

    const nextEigenvalue = rayleighDenominator <= 1e-20 ? 0 : rayleighNumerator / rayleighDenominator;
    if (Math.abs(nextEigenvalue - eigenvalue) <= 1e-10) {
      return Math.max(1, nextEigenvalue);
    }
    eigenvalue = nextEigenvalue;
    vector = next;
  }

  return Math.max(1, eigenvalue);
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

function residualMatrix(X: Matrix, code: Matrix, dictionary: Matrix): Matrix {
  const reconstruction = matrixMultiply(code, dictionary);
  const out: Matrix = new Array(X.length);
  for (let i = 0; i < X.length; i += 1) {
    const row = new Array<number>(X[0].length);
    for (let j = 0; j < X[0].length; j += 1) {
      row[j] = X[i][j] - reconstruction[i][j];
    }
    out[i] = row;
  }
  return out;
}

function meanSquaredError(matrix: Matrix): number {
  let sum = 0;
  const rows = matrix.length;
  const cols = matrix[0].length;
  for (let i = 0; i < rows; i += 1) {
    for (let j = 0; j < cols; j += 1) {
      sum += matrix[i][j] * matrix[i][j];
    }
  }
  return sum / (rows * cols);
}

export class DictionaryLearning {
  components_: Matrix | null = null;
  error_: number | null = null;
  nFeaturesIn_: number | null = null;
  nComponents_: number | null = null;
  mean_: Vector | null = null;

  protected nComponents?: number;
  protected alpha: number;
  protected maxIter: number;
  protected tolerance: number;
  protected randomState?: number;
  protected transformAlpha: number;
  protected fitted = false;

  constructor(options: DictionaryLearningOptions = {}) {
    this.nComponents = options.nComponents;
    this.alpha = options.alpha ?? 1;
    this.maxIter = options.maxIter ?? 1000;
    this.tolerance = options.tolerance ?? 1e-8;
    this.randomState = options.randomState;
    this.transformAlpha = options.transformAlpha ?? this.alpha;

    if (this.nComponents !== undefined && (!Number.isInteger(this.nComponents) || this.nComponents < 1)) {
      throw new Error(`nComponents must be an integer >= 1 when provided. Got ${this.nComponents}.`);
    }
    if (!Number.isFinite(this.alpha) || this.alpha < 0) {
      throw new Error(`alpha must be finite and >= 0. Got ${this.alpha}.`);
    }
    if (!Number.isInteger(this.maxIter) || this.maxIter < 1) {
      throw new Error(`maxIter must be an integer >= 1. Got ${this.maxIter}.`);
    }
    if (!Number.isFinite(this.tolerance) || this.tolerance < 0) {
      throw new Error(`tolerance must be finite and >= 0. Got ${this.tolerance}.`);
    }
    if (!Number.isFinite(this.transformAlpha) || this.transformAlpha < 0) {
      throw new Error(`transformAlpha must be finite and >= 0. Got ${this.transformAlpha}.`);
    }
  }

  fit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    const nFeatures = X[0].length;
    const nComponents = this.resolveNComponents(nFeatures);
    const rng = this.randomState === undefined ? Math.random : mulberry32(this.randomState);

    const mean = computeFeatureMeans(X);
    const centered = centerMatrix(X, mean);
    const dictionary = this.initializeDictionary(centered, nComponents, rng);

    let previousError = Number.POSITIVE_INFINITY;
    let bestError = previousError;

    for (let iter = 0; iter < this.maxIter; iter += 1) {
      const code = this.encodeSparse(centered, dictionary, this.alpha);
      const residual = residualMatrix(centered, code, dictionary);
      const mse = meanSquaredError(residual);
      bestError = mse;

      this.updateDictionary(centered, code, dictionary, rng);
      if (Math.abs(previousError - mse) <= this.tolerance) {
        break;
      }
      previousError = mse;
    }

    this.components_ = dictionary.map((row) => row.slice());
    this.mean_ = mean;
    this.nFeaturesIn_ = nFeatures;
    this.nComponents_ = nComponents;
    this.error_ = Number.isFinite(bestError) ? bestError : 0;
    this.fitted = true;
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

    const centered = centerMatrix(X, this.mean_!);
    return this.encodeSparse(centered, this.components_!, this.transformAlpha);
  }

  fitTransform(X: Matrix): Matrix {
    return this.fit(X).transform(X);
  }

  inverseTransform(code: Matrix): Matrix {
    this.assertFitted();
    assertNonEmptyMatrix(code);
    assertConsistentRowSize(code);
    assertFiniteMatrix(code);
    if (code[0].length !== this.components_!.length) {
      throw new Error(
        `Code width mismatch. Expected ${this.components_!.length}, got ${code[0].length}.`,
      );
    }

    const out = matrixMultiply(code, this.components_!);
    for (let i = 0; i < out.length; i += 1) {
      for (let j = 0; j < out[i].length; j += 1) {
        out[i][j] += this.mean_![j];
      }
    }
    return out;
  }

  protected resolveNComponents(nFeatures: number): number {
    return this.nComponents ?? nFeatures;
  }

  protected initializeDictionary(
    centeredX: Matrix,
    nComponents: number,
    random: () => number,
  ): Matrix {
    const nSamples = centeredX.length;
    const nFeatures = centeredX[0].length;
    const dictionary: Matrix = new Array(nComponents);
    for (let k = 0; k < nComponents; k += 1) {
      const source = centeredX[Math.floor(random() * nSamples)];
      let atom = normalizeRow(source);
      let norm = 0;
      for (let j = 0; j < atom.length; j += 1) {
        norm += atom[j] * atom[j];
      }
      if (norm <= 1e-12) {
        atom = new Array<number>(nFeatures);
        for (let j = 0; j < nFeatures; j += 1) {
          atom[j] = random() * 2 - 1;
        }
        atom = normalizeRow(atom);
      }
      dictionary[k] = atom;
    }
    return dictionary;
  }

  protected encodeSparse(XCentered: Matrix, dictionary: Matrix, alpha: number): Matrix {
    const nSamples = XCentered.length;
    const nComponents = dictionary.length;
    const gram = gramMatrix(dictionary);
    const lipschitz = largestEigenvalueSymmetric(gram);
    const step = 1 / Math.max(lipschitz, 1e-8);
    const l1Step = alpha * step;

    const codes: Matrix = new Array(nSamples);
    for (let i = 0; i < nSamples; i += 1) {
      const b = new Array<number>(nComponents).fill(0);
      for (let k = 0; k < nComponents; k += 1) {
        b[k] = dot(dictionary[k], XCentered[i]);
      }

      const code = new Array<number>(nComponents).fill(0);
      for (let iter = 0; iter < 64; iter += 1) {
        for (let k = 0; k < nComponents; k += 1) {
          let gradient = -b[k];
          const gramRow = gram[k];
          for (let j = 0; j < nComponents; j += 1) {
            gradient += gramRow[j] * code[j];
          }
          code[k] = softThreshold(code[k] - step * gradient, l1Step);
        }
      }
      codes[i] = code;
    }

    return codes;
  }

  protected updateDictionary(
    centeredX: Matrix,
    code: Matrix,
    dictionary: Matrix,
    random: () => number,
  ): void {
    const nSamples = centeredX.length;
    const nFeatures = centeredX[0].length;
    const residual = residualMatrix(centeredX, code, dictionary);

    for (let component = 0; component < dictionary.length; component += 1) {
      const oldAtom = dictionary[component];
      const coefficients = new Array<number>(nSamples);
      let coefficientNorm = 0;
      for (let i = 0; i < nSamples; i += 1) {
        const value = code[i][component];
        coefficients[i] = value;
        coefficientNorm += value * value;
      }

      if (coefficientNorm <= 1e-12) {
        const source = centeredX[Math.floor(random() * nSamples)];
        dictionary[component] = normalizeRow(source);
        continue;
      }

      const restoredResidual: Matrix = new Array(nSamples);
      for (let i = 0; i < nSamples; i += 1) {
        const row = new Array<number>(nFeatures);
        const coeff = coefficients[i];
        for (let j = 0; j < nFeatures; j += 1) {
          row[j] = residual[i][j] + coeff * oldAtom[j];
        }
        restoredResidual[i] = row;
      }

      const updatedAtom = new Array<number>(nFeatures).fill(0);
      for (let i = 0; i < nSamples; i += 1) {
        const coeff = coefficients[i];
        for (let j = 0; j < nFeatures; j += 1) {
          updatedAtom[j] += coeff * restoredResidual[i][j];
        }
      }
      for (let j = 0; j < nFeatures; j += 1) {
        updatedAtom[j] /= coefficientNorm;
      }
      dictionary[component] = normalizeRow(updatedAtom);

      const newAtom = dictionary[component];
      for (let i = 0; i < nSamples; i += 1) {
        const coeff = coefficients[i];
        for (let j = 0; j < nFeatures; j += 1) {
          residual[i][j] = restoredResidual[i][j] - coeff * newAtom[j];
        }
      }
    }
  }

  protected finalizeFit(
    dictionary: Matrix,
    mean: Vector,
    error: number,
    nFeatures: number,
  ): void {
    this.components_ = dictionary.map((row) => row.slice());
    this.mean_ = mean.slice();
    this.nFeaturesIn_ = nFeatures;
    this.nComponents_ = dictionary.length;
    this.error_ = error;
    this.fitted = true;
  }

  private assertFitted(): void {
    if (!this.fitted || !this.components_ || !this.mean_ || this.nFeaturesIn_ === null) {
      throw new Error("DictionaryLearning has not been fitted.");
    }
  }
}
