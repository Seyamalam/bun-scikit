import type { Matrix, Vector } from "../types";

export function assertNonEmptyMatrix(X: Matrix, label = "X"): void {
  if (!Array.isArray(X) || X.length === 0) {
    throw new Error(`${label} must be a non-empty 2D array.`);
  }

  if (!Array.isArray(X[0]) || X[0].length === 0) {
    throw new Error(`${label} must have at least one feature column.`);
  }
}

export function assertConsistentRowSize(X: Matrix, label = "X"): void {
  const featureCount = X[0]?.length ?? 0;
  for (let rowIndex = 0; rowIndex < X.length; rowIndex += 1) {
    const row = X[rowIndex];
    if (!Array.isArray(row) || row.length !== featureCount) {
      throw new Error(
        `${label} rows must all have the same length. Row ${rowIndex} differs.`,
      );
    }
  }
}

export function assertFiniteMatrix(X: Matrix, label = "X"): void {
  for (let i = 0; i < X.length; i += 1) {
    for (let j = 0; j < X[i].length; j += 1) {
      const value = X[i][j];
      if (!Number.isFinite(value)) {
        throw new Error(`${label} contains a non-finite value at [${i}, ${j}].`);
      }
    }
  }
}

export function assertVectorLength(
  y: Vector,
  expectedLength: number,
  label = "y",
): void {
  if (!Array.isArray(y) || y.length !== expectedLength) {
    throw new Error(`${label} length must equal ${expectedLength}.`);
  }
}

export function assertFiniteVector(y: Vector, label = "y"): void {
  for (let i = 0; i < y.length; i += 1) {
    if (!Number.isFinite(y[i])) {
      throw new Error(`${label} contains a non-finite value at index ${i}.`);
    }
  }
}

export function validateRegressionInputs(X: Matrix, y: Vector): void {
  assertNonEmptyMatrix(X);
  assertConsistentRowSize(X);
  assertFiniteMatrix(X);
  assertVectorLength(y, X.length);
  assertFiniteVector(y);
}

export function assertBinaryVector(y: Vector, label = "y"): void {
  for (let i = 0; i < y.length; i += 1) {
    const value = y[i];
    if (!(value === 0 || value === 1)) {
      throw new Error(`${label} must be binary (0 or 1). Found ${value} at index ${i}.`);
    }
  }
}

export function validateClassificationInputs(X: Matrix, y: Vector): void {
  assertNonEmptyMatrix(X);
  assertConsistentRowSize(X);
  assertFiniteMatrix(X);
  assertVectorLength(y, X.length);
  assertFiniteVector(y);
}

export function validateBinaryClassificationInputs(X: Matrix, y: Vector): void {
  validateClassificationInputs(X, y);
  assertBinaryVector(y);
}
