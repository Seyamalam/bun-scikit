import type { Matrix, Vector } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  assertNonEmptyMatrix,
} from "../utils/validation";

export function validateMultiOutputInputs(X: Matrix, Y: Matrix): void {
  assertNonEmptyMatrix(X);
  assertConsistentRowSize(X);
  assertFiniteMatrix(X);
  assertNonEmptyMatrix(Y, "Y");
  assertConsistentRowSize(Y, "Y");
  assertFiniteMatrix(Y, "Y");
  if (X.length !== Y.length) {
    throw new Error(`Y must have the same number of rows as X. Expected ${X.length}, got ${Y.length}.`);
  }
}

export function validateSampleWeight(sampleWeight: Vector | undefined, nSamples: number): void {
  if (!sampleWeight) {
    return;
  }
  if (sampleWeight.length !== nSamples) {
    throw new Error(`sampleWeight length must match sample count. Got ${sampleWeight.length} and ${nSamples}.`);
  }
  assertFiniteVector(sampleWeight, "sampleWeight");
}

export function extractColumn(Y: Matrix, columnIndex: number): Vector {
  const column = new Array<number>(Y.length);
  for (let i = 0; i < Y.length; i += 1) {
    column[i] = Y[i][columnIndex];
  }
  return column;
}

export function setColumn(Y: Matrix, columnIndex: number, values: Vector): void {
  for (let i = 0; i < Y.length; i += 1) {
    Y[i][columnIndex] = values[i];
  }
}

export function emptyPredictionMatrix(nSamples: number, nOutputs: number): Matrix {
  return Array.from({ length: nSamples }, () => new Array<number>(nOutputs).fill(0));
}

function defaultCloneEstimator<T>(estimator: T): T {
  const instance = estimator as unknown as {
    constructor: new (options?: Record<string, unknown>) => T;
    getParams?: (deep?: boolean) => Record<string, unknown>;
  };
  if (typeof instance.constructor !== "function") {
    throw new Error("Estimator instance cannot be cloned. Pass a factory function instead.");
  }
  if (typeof instance.getParams === "function") {
    return new instance.constructor(instance.getParams(true));
  }
  return new instance.constructor();
}

export function resolveEstimatorClone<T>(
  estimatorFactory: (() => T) | T,
): T {
  if (typeof estimatorFactory === "function") {
    return (estimatorFactory as () => T)();
  }
  return defaultCloneEstimator(estimatorFactory);
}

export function exactMatchAccuracy(yTrue: Matrix, yPred: Matrix): number {
  let correct = 0;
  for (let i = 0; i < yTrue.length; i += 1) {
    let allEqual = true;
    for (let j = 0; j < yTrue[i].length; j += 1) {
      if (yTrue[i][j] !== yPred[i][j]) {
        allEqual = false;
        break;
      }
    }
    if (allEqual) {
      correct += 1;
    }
  }
  return correct / yTrue.length;
}

export function augmentWithColumns(
  X: Matrix,
  Y: Matrix,
  columnOrder: number[],
): Matrix {
  const out: Matrix = new Array(X.length);
  for (let i = 0; i < X.length; i += 1) {
    const row = X[i].slice();
    for (let j = 0; j < columnOrder.length; j += 1) {
      row.push(Y[i][columnOrder[j]]);
    }
    out[i] = row;
  }
  return out;
}

export function randomPermutation(length: number, seed: number): number[] {
  let state = seed >>> 0;
  function next(): number {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state ^ (state >>> 15);
    t = Math.imul(t, state | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  const out = Array.from({ length }, (_, index) => index);
  for (let i = out.length - 1; i > 0; i -= 1) {
    const j = Math.floor(next() * (i + 1));
    const tmp = out[i];
    out[i] = out[j];
    out[j] = tmp;
  }
  return out;
}
