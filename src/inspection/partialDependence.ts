import type { Matrix, Vector } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";

export interface PartialDependenceEstimator {
  predict(X: Matrix): Vector;
  predictProba?: (X: Matrix) => Matrix;
}

export type PartialDependenceResponseMethod = "auto" | "predict" | "predict_proba";

export interface PartialDependenceOptions {
  features: number[];
  gridResolution?: number;
  percentiles?: [number, number];
  kind?: "average" | "individual" | "both";
  responseMethod?: PartialDependenceResponseMethod;
  target?: number;
}

export interface PartialDependenceResult {
  features: number[];
  values: Vector[];
  average: Matrix;
  individual?: number[][][];
}

function percentile(values: Vector, q: number): number {
  if (values.length === 0) {
    throw new Error("Cannot compute percentile of an empty vector.");
  }
  const sorted = values.slice().sort((a, b) => a - b);
  const position = Math.min(sorted.length - 1, Math.max(0, q * (sorted.length - 1)));
  const low = Math.floor(position);
  const high = Math.ceil(position);
  if (low === high) {
    return sorted[low];
  }
  const ratio = position - low;
  return sorted[low] * (1 - ratio) + sorted[high] * ratio;
}

function linspace(start: number, stop: number, size: number): Vector {
  if (size <= 1) {
    return [start];
  }
  const out = new Array<number>(size);
  const step = (stop - start) / (size - 1);
  for (let i = 0; i < size; i += 1) {
    out[i] = start + step * i;
  }
  return out;
}

function cloneMatrix(X: Matrix): Matrix {
  return X.map((row) => row.slice());
}

function getResponse(
  estimator: PartialDependenceEstimator,
  X: Matrix,
  method: PartialDependenceResponseMethod,
  target: number,
): Vector {
  if (method === "predict") {
    return estimator.predict(X);
  }
  if (method === "predict_proba") {
    if (typeof estimator.predictProba !== "function") {
      throw new Error("responseMethod='predict_proba' requires estimator.predictProba().");
    }
    const proba = estimator.predictProba(X);
    const classIndex = Math.max(0, Math.min(target, proba[0].length - 1));
    return proba.map((row) => row[classIndex]);
  }
  if (typeof estimator.predictProba === "function") {
    const proba = estimator.predictProba(X);
    const classIndex = Math.max(0, Math.min(target, proba[0].length - 1));
    return proba.map((row) => row[classIndex]);
  }
  return estimator.predict(X);
}

export function partialDependence(
  estimator: PartialDependenceEstimator,
  X: Matrix,
  options: PartialDependenceOptions,
): PartialDependenceResult {
  if (typeof estimator !== "object" || estimator === null) {
    throw new Error("estimator must be an object with predict().");
  }
  if (typeof estimator.predict !== "function") {
    throw new Error("estimator must implement predict().");
  }

  assertNonEmptyMatrix(X);
  assertConsistentRowSize(X);
  assertFiniteMatrix(X);

  const features = options.features;
  if (!Array.isArray(features) || features.length === 0) {
    throw new Error("features must be a non-empty array of feature indices.");
  }
  for (let i = 0; i < features.length; i += 1) {
    const featureIndex = features[i];
    if (!Number.isInteger(featureIndex) || featureIndex < 0 || featureIndex >= X[0].length) {
      throw new Error(`Invalid feature index ${featureIndex}.`);
    }
  }

  const gridResolution = options.gridResolution ?? 20;
  if (!Number.isInteger(gridResolution) || gridResolution < 2) {
    throw new Error(`gridResolution must be an integer >= 2. Got ${gridResolution}.`);
  }

  const percentiles = options.percentiles ?? [0.05, 0.95];
  if (
    percentiles[0] < 0 ||
    percentiles[1] > 1 ||
    percentiles[0] >= percentiles[1]
  ) {
    throw new Error("percentiles must be within [0, 1] and strictly increasing.");
  }

  const kind = options.kind ?? "average";
  const responseMethod = options.responseMethod ?? "auto";
  const target = options.target ?? 1;

  const values: Vector[] = [];
  const average: Matrix = [];
  const individual: number[][][] = [];

  for (let featureCursor = 0; featureCursor < features.length; featureCursor += 1) {
    const featureIndex = features[featureCursor];
    const column = X.map((row) => row[featureIndex]);
    const minValue = percentile(column, percentiles[0]);
    const maxValue = percentile(column, percentiles[1]);
    const grid = linspace(minValue, maxValue, gridResolution);
    values.push(grid);

    const avg = new Array<number>(grid.length).fill(0);
    const ind = Array.from({ length: X.length }, () => new Array<number>(grid.length).fill(0));
    for (let gridIndex = 0; gridIndex < grid.length; gridIndex += 1) {
      const probe = cloneMatrix(X);
      for (let rowIndex = 0; rowIndex < probe.length; rowIndex += 1) {
        probe[rowIndex][featureIndex] = grid[gridIndex];
      }
      const response = getResponse(estimator, probe, responseMethod, target);
      let total = 0;
      for (let rowIndex = 0; rowIndex < response.length; rowIndex += 1) {
        total += response[rowIndex];
        ind[rowIndex][gridIndex] = response[rowIndex];
      }
      avg[gridIndex] = total / response.length;
    }
    average.push(avg);
    individual.push(ind);
  }

  return {
    features: features.slice(),
    values,
    average,
    individual: kind === "individual" || kind === "both" ? individual : undefined,
  };
}
