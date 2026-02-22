import type { Matrix, Vector } from "../types";

export function transpose(X: Matrix): Matrix {
  const rows = X.length;
  const cols = X[0].length;
  const result: Matrix = Array.from({ length: cols }, () =>
    new Array(rows).fill(0),
  );

  for (let i = 0; i < rows; i += 1) {
    for (let j = 0; j < cols; j += 1) {
      result[j][i] = X[i][j];
    }
  }

  return result;
}

export function multiplyMatrices(A: Matrix, B: Matrix): Matrix {
  const aRows = A.length;
  const aCols = A[0].length;
  const bRows = B.length;
  const bCols = B[0].length;

  if (aCols !== bRows) {
    throw new Error(
      `Matrix dimensions do not align: ${aRows}x${aCols} times ${bRows}x${bCols}.`,
    );
  }

  const result: Matrix = Array.from({ length: aRows }, () =>
    new Array(bCols).fill(0),
  );

  for (let i = 0; i < aRows; i += 1) {
    for (let k = 0; k < aCols; k += 1) {
      const aik = A[i][k];
      for (let j = 0; j < bCols; j += 1) {
        result[i][j] += aik * B[k][j];
      }
    }
  }

  return result;
}

export function multiplyMatrixVector(A: Matrix, x: Vector): Vector {
  const rows = A.length;
  const cols = A[0].length;

  if (cols !== x.length) {
    throw new Error(
      `Matrix-vector dimensions do not align: ${rows}x${cols} times ${x.length}.`,
    );
  }

  const result = new Array(rows).fill(0);
  for (let i = 0; i < rows; i += 1) {
    let sum = 0;
    for (let j = 0; j < cols; j += 1) {
      sum += A[i][j] * x[j];
    }
    result[i] = sum;
  }

  return result;
}

export function addInterceptColumn(X: Matrix): Matrix {
  return X.map((row) => [1, ...row]);
}

export function identityMatrix(size: number): Matrix {
  const I: Matrix = Array.from({ length: size }, () => new Array(size).fill(0));
  for (let i = 0; i < size; i += 1) {
    I[i][i] = 1;
  }
  return I;
}

export function inverseMatrix(A: Matrix): Matrix {
  const n = A.length;
  if (n === 0 || A[0].length !== n) {
    throw new Error("Only non-empty square matrices can be inverted.");
  }

  const EPSILON = 1e-12;
  const augmented: Matrix = A.map((row, i) => [...row, ...identityMatrix(n)[i]]);

  for (let col = 0; col < n; col += 1) {
    let pivotRow = col;
    let maxAbs = Math.abs(augmented[pivotRow][col]);

    for (let r = col + 1; r < n; r += 1) {
      const value = Math.abs(augmented[r][col]);
      if (value > maxAbs) {
        maxAbs = value;
        pivotRow = r;
      }
    }

    if (maxAbs < EPSILON) {
      throw new Error("Matrix is singular or near-singular and cannot be inverted.");
    }

    if (pivotRow !== col) {
      const tmp = augmented[col];
      augmented[col] = augmented[pivotRow];
      augmented[pivotRow] = tmp;
    }

    const pivot = augmented[col][col];
    for (let j = 0; j < 2 * n; j += 1) {
      augmented[col][j] /= pivot;
    }

    for (let r = 0; r < n; r += 1) {
      if (r === col) {
        continue;
      }

      const factor = augmented[r][col];
      if (factor === 0) {
        continue;
      }

      for (let j = 0; j < 2 * n; j += 1) {
        augmented[r][j] -= factor * augmented[col][j];
      }
    }
  }

  return augmented.map((row) => row.slice(n));
}

export function solveSymmetricPositiveDefinite(A: Matrix, b: Vector): Vector {
  const n = A.length;
  if (n === 0 || A[0].length !== n) {
    throw new Error("A must be a non-empty square matrix.");
  }
  if (b.length !== n) {
    throw new Error(`b length must match matrix size ${n}. Got ${b.length}.`);
  }

  const L: Matrix = Array.from({ length: n }, () => new Array(n).fill(0));
  const EPSILON = 1e-12;

  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j <= i; j += 1) {
      let sum = A[i][j];
      for (let k = 0; k < j; k += 1) {
        sum -= L[i][k] * L[j][k];
      }

      if (i === j) {
        if (sum <= EPSILON) {
          throw new Error("Matrix is not positive definite.");
        }
        L[i][j] = Math.sqrt(sum);
      } else {
        L[i][j] = sum / L[j][j];
      }
    }
  }

  const y = new Array(n).fill(0);
  for (let i = 0; i < n; i += 1) {
    let sum = b[i];
    for (let k = 0; k < i; k += 1) {
      sum -= L[i][k] * y[k];
    }
    y[i] = sum / L[i][i];
  }

  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i -= 1) {
    let sum = y[i];
    for (let k = i + 1; k < n; k += 1) {
      sum -= L[k][i] * x[k];
    }
    x[i] = sum / L[i][i];
  }

  return x;
}

export function dot(a: Vector, b: Vector): number {
  if (a.length !== b.length) {
    throw new Error(`Vector sizes do not match: ${a.length} vs ${b.length}.`);
  }

  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    sum += a[i] * b[i];
  }
  return sum;
}

export function mean(values: Vector): number {
  if (values.length === 0) {
    throw new Error("Cannot compute mean of an empty vector.");
  }

  let total = 0;
  for (let i = 0; i < values.length; i += 1) {
    total += values[i];
  }
  return total / values.length;
}
