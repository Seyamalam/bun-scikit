import type { Matrix, Vector } from "../types";
import { dot, identityMatrix } from "../utils/linalg";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";

export interface PCAOptions {
  nComponents?: number;
  whiten?: boolean;
  tolerance?: number;
  maxIter?: number;
}

interface EigenDecomposition {
  eigenvalues: Vector;
  eigenvectors: Matrix;
}

function cloneMatrix(X: Matrix): Matrix {
  return X.map((row) => row.slice());
}

function jacobiEigenDecomposition(
  covariance: Matrix,
  tolerance: number,
  maxIter: number,
): EigenDecomposition {
  const size = covariance.length;
  const A = cloneMatrix(covariance);
  const V = identityMatrix(size);

  for (let iter = 0; iter < maxIter; iter += 1) {
    let p = 0;
    let q = 1;
    let maxOffDiagonal = 0;

    for (let i = 0; i < size; i += 1) {
      for (let j = i + 1; j < size; j += 1) {
        const absValue = Math.abs(A[i][j]);
        if (absValue > maxOffDiagonal) {
          maxOffDiagonal = absValue;
          p = i;
          q = j;
        }
      }
    }

    if (maxOffDiagonal <= tolerance) {
      break;
    }

    const app = A[p][p];
    const aqq = A[q][q];
    const apq = A[p][q];

    const phi = 0.5 * Math.atan2(2 * apq, aqq - app);
    const c = Math.cos(phi);
    const s = Math.sin(phi);

    for (let i = 0; i < size; i += 1) {
      if (i === p || i === q) {
        continue;
      }
      const aip = A[i][p];
      const aiq = A[i][q];
      const rotatedIP = c * aip - s * aiq;
      const rotatedIQ = s * aip + c * aiq;
      A[i][p] = rotatedIP;
      A[p][i] = rotatedIP;
      A[i][q] = rotatedIQ;
      A[q][i] = rotatedIQ;
    }

    const appRotated = c * c * app - 2 * s * c * apq + s * s * aqq;
    const aqqRotated = s * s * app + 2 * s * c * apq + c * c * aqq;
    A[p][p] = appRotated;
    A[q][q] = aqqRotated;
    A[p][q] = 0;
    A[q][p] = 0;

    for (let i = 0; i < size; i += 1) {
      const vip = V[i][p];
      const viq = V[i][q];
      V[i][p] = c * vip - s * viq;
      V[i][q] = s * vip + c * viq;
    }
  }

  for (let column = 0; column < size; column += 1) {
    let squaredNorm = 0;
    for (let row = 0; row < size; row += 1) {
      squaredNorm += V[row][column] * V[row][column];
    }
    const norm = Math.sqrt(squaredNorm);
    if (norm === 0) {
      continue;
    }
    for (let row = 0; row < size; row += 1) {
      V[row][column] /= norm;
    }
  }

  const eigenvalues = new Array<number>(size).fill(0);
  for (let i = 0; i < size; i += 1) {
    eigenvalues[i] = A[i][i];
  }

  return { eigenvalues, eigenvectors: V };
}

function computeFeatureMeans(X: Matrix): Vector {
  const nSamples = X.length;
  const nFeatures = X[0].length;
  const means = new Array<number>(nFeatures).fill(0);
  for (let i = 0; i < nSamples; i += 1) {
    for (let j = 0; j < nFeatures; j += 1) {
      means[j] += X[i][j];
    }
  }
  for (let j = 0; j < nFeatures; j += 1) {
    means[j] /= nSamples;
  }
  return means;
}

function centerMatrix(X: Matrix, means: Vector): Matrix {
  const centered: Matrix = new Array(X.length);
  for (let i = 0; i < X.length; i += 1) {
    const row = new Array<number>(X[0].length);
    for (let j = 0; j < X[0].length; j += 1) {
      row[j] = X[i][j] - means[j];
    }
    centered[i] = row;
  }
  return centered;
}

function covarianceMatrix(centeredX: Matrix): Matrix {
  const nSamples = centeredX.length;
  const nFeatures = centeredX[0].length;
  const denominator = nSamples > 1 ? nSamples - 1 : 1;
  const covariance: Matrix = Array.from({ length: nFeatures }, () =>
    new Array(nFeatures).fill(0),
  );

  for (let i = 0; i < nFeatures; i += 1) {
    for (let j = i; j < nFeatures; j += 1) {
      let sum = 0;
      for (let sampleIndex = 0; sampleIndex < nSamples; sampleIndex += 1) {
        sum += centeredX[sampleIndex][i] * centeredX[sampleIndex][j];
      }
      const cov = sum / denominator;
      covariance[i][j] = cov;
      covariance[j][i] = cov;
    }
  }

  return covariance;
}

export class PCA {
  components_: Matrix | null = null;
  explainedVariance_: Vector | null = null;
  explainedVarianceRatio_: Vector | null = null;
  singularValues_: Vector | null = null;
  mean_: Vector | null = null;
  nFeaturesIn_: number | null = null;

  private readonly nComponents?: number;
  private readonly whiten: boolean;
  private readonly tolerance: number;
  private readonly maxIter: number;
  private isFitted = false;

  constructor(options: PCAOptions = {}) {
    this.nComponents = options.nComponents;
    this.whiten = options.whiten ?? false;
    this.tolerance = options.tolerance ?? 1e-12;
    this.maxIter = options.maxIter ?? 10_000;

    if (this.nComponents !== undefined && (!Number.isInteger(this.nComponents) || this.nComponents < 1)) {
      throw new Error(`nComponents must be an integer >= 1 when provided. Got ${this.nComponents}.`);
    }
    if (!Number.isFinite(this.tolerance) || this.tolerance <= 0) {
      throw new Error(`tolerance must be finite and > 0. Got ${this.tolerance}.`);
    }
    if (!Number.isInteger(this.maxIter) || this.maxIter < 1) {
      throw new Error(`maxIter must be an integer >= 1. Got ${this.maxIter}.`);
    }
  }

  fit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    const nSamples = X.length;
    const nFeatures = X[0].length;
    const maxComponents = Math.min(nSamples, nFeatures);
    const selectedComponents = this.nComponents ?? maxComponents;
    if (selectedComponents > maxComponents) {
      throw new Error(
        `nComponents (${selectedComponents}) cannot exceed min(nSamples, nFeatures) (${maxComponents}).`,
      );
    }

    const means = computeFeatureMeans(X);
    const centered = centerMatrix(X, means);
    const covariance = covarianceMatrix(centered);
    const decomposition = jacobiEigenDecomposition(covariance, this.tolerance, this.maxIter);

    const order = Array.from({ length: decomposition.eigenvalues.length }, (_, idx) => idx).sort(
      (a, b) => decomposition.eigenvalues[b] - decomposition.eigenvalues[a],
    );

    const explainedVariance = new Array<number>(selectedComponents).fill(0);
    const components: Matrix = new Array(selectedComponents);
    const totalVariance = decomposition.eigenvalues.reduce(
      (acc, value) => acc + Math.max(0, value),
      0,
    );

    for (let componentIndex = 0; componentIndex < selectedComponents; componentIndex += 1) {
      const eigenIndex = order[componentIndex];
      const eigenvalue = Math.max(0, decomposition.eigenvalues[eigenIndex]);
      explainedVariance[componentIndex] = eigenvalue;
      const component = new Array<number>(nFeatures);
      for (let featureIndex = 0; featureIndex < nFeatures; featureIndex += 1) {
        component[featureIndex] = decomposition.eigenvectors[featureIndex][eigenIndex];
      }
      components[componentIndex] = component;
    }

    const explainedVarianceRatio = explainedVariance.map((value) =>
      totalVariance === 0 ? 0 : value / totalVariance,
    );
    const singularValues = explainedVariance.map((value) =>
      Math.sqrt(Math.max(0, value * (nSamples > 1 ? nSamples - 1 : 1))),
    );

    this.components_ = components;
    this.explainedVariance_ = explainedVariance;
    this.explainedVarianceRatio_ = explainedVarianceRatio;
    this.singularValues_ = singularValues;
    this.mean_ = means;
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

    const transformed: Matrix = new Array(X.length);
    for (let sampleIndex = 0; sampleIndex < X.length; sampleIndex += 1) {
      const centered = new Array<number>(this.nFeaturesIn_!);
      for (let featureIndex = 0; featureIndex < this.nFeaturesIn_!; featureIndex += 1) {
        centered[featureIndex] = X[sampleIndex][featureIndex] - this.mean_![featureIndex];
      }
      const projection = new Array<number>(this.components_!.length);
      for (let componentIndex = 0; componentIndex < this.components_!.length; componentIndex += 1) {
        let value = dot(centered, this.components_![componentIndex]);
        if (this.whiten) {
          value /= Math.sqrt(Math.max(this.explainedVariance_![componentIndex], 1e-12));
        }
        projection[componentIndex] = value;
      }
      transformed[sampleIndex] = projection;
    }
    return transformed;
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

    const reconstructed: Matrix = new Array(X.length);
    for (let sampleIndex = 0; sampleIndex < X.length; sampleIndex += 1) {
      const row = new Array<number>(this.nFeaturesIn_!);
      for (let featureIndex = 0; featureIndex < this.nFeaturesIn_!; featureIndex += 1) {
        row[featureIndex] = this.mean_![featureIndex];
      }

      for (let componentIndex = 0; componentIndex < this.components_!.length; componentIndex += 1) {
        const scaled =
          this.whiten
            ? X[sampleIndex][componentIndex] *
              Math.sqrt(Math.max(this.explainedVariance_![componentIndex], 1e-12))
            : X[sampleIndex][componentIndex];
        const component = this.components_![componentIndex];
        for (let featureIndex = 0; featureIndex < this.nFeaturesIn_!; featureIndex += 1) {
          row[featureIndex] += scaled * component[featureIndex];
        }
      }

      reconstructed[sampleIndex] = row;
    }

    return reconstructed;
  }

  private assertFitted(): void {
    if (
      !this.isFitted ||
      !this.components_ ||
      !this.explainedVariance_ ||
      !this.mean_ ||
      this.nFeaturesIn_ === null
    ) {
      throw new Error("PCA has not been fitted.");
    }
  }
}
