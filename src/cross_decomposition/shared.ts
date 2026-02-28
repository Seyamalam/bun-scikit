import type { Matrix, Vector } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  assertNonEmptyMatrix,
} from "../utils/validation";
import { dot, inverseMatrix, multiplyMatrices, transpose } from "../utils/linalg";

export interface CrossDecompositionInput {
  Y: Matrix;
  targetIsVector: boolean;
}

export interface CenterScaleResult {
  transformed: Matrix;
  mean: Vector;
  scale: Vector;
}

export interface TopSingularVectors {
  left: Matrix;
  right: Matrix;
  singularValues: Vector;
}

export function toTargetMatrix(Y: Matrix | Vector): CrossDecompositionInput {
  if (!Array.isArray(Y) || Y.length === 0) {
    throw new Error("Y must be a non-empty vector or matrix.");
  }
  if (Array.isArray(Y[0])) {
    const asMatrix = Y as Matrix;
    assertNonEmptyMatrix(asMatrix, "Y");
    assertConsistentRowSize(asMatrix, "Y");
    assertFiniteMatrix(asMatrix, "Y");
    return { Y: asMatrix.map((row) => row.slice()), targetIsVector: false };
  }
  const asVector = Y as Vector;
  assertFiniteVector(asVector, "Y");
  const matrix = asVector.map((value) => [value]);
  return { Y: matrix, targetIsVector: true };
}

export function validateCrossDecompositionInputs(X: Matrix, Y: Matrix): void {
  assertNonEmptyMatrix(X);
  assertConsistentRowSize(X);
  assertFiniteMatrix(X);
  assertNonEmptyMatrix(Y, "Y");
  assertConsistentRowSize(Y, "Y");
  assertFiniteMatrix(Y, "Y");
  if (Y.length !== X.length) {
    throw new Error(`Y must have the same number of rows as X. Expected ${X.length}, got ${Y.length}.`);
  }
}

export function centerAndScale(X: Matrix, scale: boolean): CenterScaleResult {
  const nSamples = X.length;
  const nFeatures = X[0].length;
  const mean = new Array<number>(nFeatures).fill(0);
  const transformed: Matrix = new Array(nSamples);

  for (let i = 0; i < nSamples; i += 1) {
    for (let j = 0; j < nFeatures; j += 1) {
      mean[j] += X[i][j];
    }
  }
  for (let j = 0; j < nFeatures; j += 1) {
    mean[j] /= nSamples;
  }

  const scaleVector = new Array<number>(nFeatures).fill(1);
  if (scale) {
    for (let j = 0; j < nFeatures; j += 1) {
      scaleVector[j] = 0;
    }
    for (let i = 0; i < nSamples; i += 1) {
      for (let j = 0; j < nFeatures; j += 1) {
        const diff = X[i][j] - mean[j];
        scaleVector[j] += diff * diff;
      }
    }
    for (let j = 0; j < nFeatures; j += 1) {
      scaleVector[j] = Math.sqrt(scaleVector[j] / Math.max(1, nSamples - 1));
      if (scaleVector[j] <= 1e-12) {
        scaleVector[j] = 1;
      }
    }
  }

  for (let i = 0; i < nSamples; i += 1) {
    const row = new Array<number>(nFeatures);
    for (let j = 0; j < nFeatures; j += 1) {
      row[j] = (X[i][j] - mean[j]) / scaleVector[j];
    }
    transformed[i] = row;
  }

  return { transformed, mean, scale: scaleVector };
}

export function normalizeVector(vector: Vector): Vector {
  const norm = Math.sqrt(dot(vector, vector));
  if (norm <= 1e-20) {
    return vector.map(() => 0);
  }
  return vector.map((value) => value / norm);
}

export function multiplyMatrixVectorSafe(matrix: Matrix, vector: Vector): Vector {
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

export function topSingularVectors(
  matrix: Matrix,
  nComponents: number,
  maxIter: number,
  tolerance: number,
): TopSingularVectors {
  const rows = matrix.length;
  const cols = matrix[0].length;
  const left: Matrix = Array.from({ length: rows }, () => new Array<number>(nComponents).fill(0));
  const right: Matrix = Array.from({ length: cols }, () => new Array<number>(nComponents).fill(0));
  const singularValues = new Array<number>(nComponents).fill(0);

  const deflated = matrix.map((row) => row.slice());
  for (let component = 0; component < nComponents; component += 1) {
    let v = new Array<number>(cols).fill(0);
    v[component % cols] = 1;
    v = normalizeVector(v);

    for (let iter = 0; iter < maxIter; iter += 1) {
      const uRaw = multiplyMatrixVectorSafe(deflated, v);
      const u = normalizeVector(uRaw);
      const vNextRaw = multiplyMatrixVectorSafe(transpose(deflated), u);
      const vNext = normalizeVector(vNextRaw);

      let delta = 0;
      for (let i = 0; i < v.length; i += 1) {
        const diff = vNext[i] - v[i];
        delta += diff * diff;
      }
      v = vNext;
      if (delta <= tolerance) {
        break;
      }
    }

    const u = normalizeVector(multiplyMatrixVectorSafe(deflated, v));
    const sigma = dot(u, multiplyMatrixVectorSafe(deflated, v));
    singularValues[component] = sigma;

    for (let i = 0; i < rows; i += 1) {
      left[i][component] = u[i];
    }
    for (let j = 0; j < cols; j += 1) {
      right[j][component] = v[j];
    }

    for (let i = 0; i < rows; i += 1) {
      for (let j = 0; j < cols; j += 1) {
        deflated[i][j] -= sigma * u[i] * v[j];
      }
    }
  }

  return { left, right, singularValues };
}

export function ridgeLeastSquares(X: Matrix, Y: Matrix, ridge = 1e-8): Matrix {
  const Xt = transpose(X);
  const XtX = multiplyMatrices(Xt, X);
  const regularized = XtX.map((row, i) =>
    row.map((value, j) => (i === j ? value + ridge : value)),
  );
  const XtY = multiplyMatrices(Xt, Y);
  return multiplyMatrices(inverseMatrix(regularized), XtY);
}

export function matrixFromVector(y: Vector): Matrix {
  return y.map((value) => [value]);
}

export function vectorFromSingleColumnMatrix(Y: Matrix): Vector {
  return Y.map((row) => row[0]);
}

export function copyMatrix(X: Matrix): Matrix {
  return X.map((row) => row.slice());
}

export function zeros(rows: number, cols: number): Matrix {
  return Array.from({ length: rows }, () => new Array<number>(cols).fill(0));
}

export function getColumn(X: Matrix, columnIndex: number): Vector {
  const column = new Array<number>(X.length);
  for (let i = 0; i < X.length; i += 1) {
    column[i] = X[i][columnIndex];
  }
  return column;
}

export function setColumn(X: Matrix, columnIndex: number, column: Vector): void {
  for (let i = 0; i < X.length; i += 1) {
    X[i][columnIndex] = column[i];
  }
}

export function squaredNorm(values: Vector): number {
  return dot(values, values);
}

export function matVecDot(X: Matrix, vector: Vector): Vector {
  const out = new Array<number>(X.length).fill(0);
  for (let i = 0; i < X.length; i += 1) {
    let sum = 0;
    for (let j = 0; j < X[i].length; j += 1) {
      sum += X[i][j] * vector[j];
    }
    out[i] = sum;
  }
  return out;
}

export function crossVector(A: Matrix, b: Vector): Vector {
  const out = new Array<number>(A[0].length).fill(0);
  for (let j = 0; j < A[0].length; j += 1) {
    let sum = 0;
    for (let i = 0; i < A.length; i += 1) {
      sum += A[i][j] * b[i];
    }
    out[j] = sum;
  }
  return out;
}

export function deflateByOuter(X: Matrix, score: Vector, loading: Vector): void {
  for (let i = 0; i < X.length; i += 1) {
    const si = score[i];
    for (let j = 0; j < X[i].length; j += 1) {
      X[i][j] -= si * loading[j];
    }
  }
}

export function trimColumns(X: Matrix, nCols: number): Matrix {
  return X.map((row) => row.slice(0, nCols));
}

export function normalizeWithMeanAndScale(X: Matrix, mean: Vector, scale: Vector): Matrix {
  const out: Matrix = new Array(X.length);
  for (let i = 0; i < X.length; i += 1) {
    const row = new Array<number>(X[0].length);
    for (let j = 0; j < X[0].length; j += 1) {
      row[j] = (X[i][j] - mean[j]) / scale[j];
    }
    out[i] = row;
  }
  return out;
}

export function denormalizeWithMeanAndScale(X: Matrix, mean: Vector, scale: Vector): Matrix {
  const out: Matrix = new Array(X.length);
  for (let i = 0; i < X.length; i += 1) {
    const row = new Array<number>(X[0].length);
    for (let j = 0; j < X[0].length; j += 1) {
      row[j] = X[i][j] * scale[j] + mean[j];
    }
    out[i] = row;
  }
  return out;
}

export function computeRotations(weights: Matrix, loadings: Matrix, ridge = 1e-8): Matrix {
  const ptw = multiplyMatrices(transpose(loadings), weights);
  const regularized = ptw.map((row, i) => row.map((value, j) => (i === j ? value + ridge : value)));
  const inv = inverseMatrix(regularized);
  return multiplyMatrices(weights, inv);
}
