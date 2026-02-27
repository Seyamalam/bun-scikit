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
export type PartialDependenceFeature = number | [number, number];

export interface PartialDependenceOptions {
  features: PartialDependenceFeature[];
  gridResolution?: number;
  percentiles?: [number, number];
  kind?: "average" | "individual" | "both";
  responseMethod?: PartialDependenceResponseMethod;
  target?: number;
}

export interface PartialDependenceResult {
  features: PartialDependenceFeature[];
  values: Array<Vector | [Vector, Vector]>;
  gridValues: Array<Vector | [Vector, Vector]>;
  average: Array<Vector | Matrix>;
  individual?: Array<Matrix | number[][][]>;
  deciles: Record<string, Vector>;
  responseMethodUsed: "predict" | "predict_proba";
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

function resolveResponseMethod(
  estimator: PartialDependenceEstimator,
  method: PartialDependenceResponseMethod,
): "predict" | "predict_proba" {
  if (method === "predict") {
    return "predict";
  }
  if (method === "predict_proba") {
    if (typeof estimator.predictProba !== "function") {
      throw new Error("responseMethod='predict_proba' requires estimator.predictProba().");
    }
    return "predict_proba";
  }
  return typeof estimator.predictProba === "function" ? "predict_proba" : "predict";
}

function getResponse(
  estimator: PartialDependenceEstimator,
  X: Matrix,
  method: "predict" | "predict_proba",
  target: number,
): Vector {
  if (method === "predict") {
    return estimator.predict(X);
  }
  const proba = estimator.predictProba!(X);
  const classIndex = Math.max(0, Math.min(target, proba[0].length - 1));
  return proba.map((row) => row[classIndex]);
}

function featureColumns(feature: PartialDependenceFeature): [number] | [number, number] {
  if (typeof feature === "number") {
    return [feature];
  }
  return [feature[0], feature[1]];
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
    throw new Error("features must be a non-empty array of feature selectors.");
  }
  for (let i = 0; i < features.length; i += 1) {
    const columns = featureColumns(features[i]);
    for (let j = 0; j < columns.length; j += 1) {
      const featureIndex = columns[j];
      if (!Number.isInteger(featureIndex) || featureIndex < 0 || featureIndex >= X[0].length) {
        throw new Error(`Invalid feature index ${featureIndex}.`);
      }
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
  const responseMethodUsed = resolveResponseMethod(estimator, options.responseMethod ?? "auto");
  const target = options.target ?? 1;

  const values: Array<Vector | [Vector, Vector]> = [];
  const average: Array<Vector | Matrix> = [];
  const individual: Array<Matrix | number[][][]> = [];
  const deciles: Record<string, Vector> = {};

  for (let featureCursor = 0; featureCursor < features.length; featureCursor += 1) {
    const feature = features[featureCursor];
    const cols = featureColumns(feature);
    const gridA = linspace(
      percentile(X.map((row) => row[cols[0]]), percentiles[0]),
      percentile(X.map((row) => row[cols[0]]), percentiles[1]),
      gridResolution,
    );
    deciles[String(cols[0])] = linspace(
      percentile(X.map((row) => row[cols[0]]), 0.1),
      percentile(X.map((row) => row[cols[0]]), 0.9),
      9,
    );

    if (cols.length === 1) {
      values.push(gridA);
      const avg = new Array<number>(gridA.length).fill(0);
      const ind = Array.from({ length: X.length }, () => new Array<number>(gridA.length).fill(0));
      for (let gridIndex = 0; gridIndex < gridA.length; gridIndex += 1) {
        const probe = cloneMatrix(X);
        for (let rowIndex = 0; rowIndex < probe.length; rowIndex += 1) {
          probe[rowIndex][cols[0]] = gridA[gridIndex];
        }
        const response = getResponse(estimator, probe, responseMethodUsed, target);
        let total = 0;
        for (let rowIndex = 0; rowIndex < response.length; rowIndex += 1) {
          total += response[rowIndex];
          ind[rowIndex][gridIndex] = response[rowIndex];
        }
        avg[gridIndex] = total / response.length;
      }
      average.push(avg);
      individual.push(ind);
      continue;
    }

    const gridB = linspace(
      percentile(X.map((row) => row[cols[1]]), percentiles[0]),
      percentile(X.map((row) => row[cols[1]]), percentiles[1]),
      gridResolution,
    );
    deciles[String(cols[1])] = linspace(
      percentile(X.map((row) => row[cols[1]]), 0.1),
      percentile(X.map((row) => row[cols[1]]), 0.9),
      9,
    );
    values.push([gridA, gridB]);

    const avg2d: Matrix = Array.from({ length: gridA.length }, () =>
      new Array<number>(gridB.length).fill(0),
    );
    const ind2d: number[][][] = Array.from({ length: X.length }, () =>
      Array.from({ length: gridA.length }, () => new Array<number>(gridB.length).fill(0)),
    );
    for (let i = 0; i < gridA.length; i += 1) {
      for (let j = 0; j < gridB.length; j += 1) {
        const probe = cloneMatrix(X);
        for (let rowIndex = 0; rowIndex < probe.length; rowIndex += 1) {
          probe[rowIndex][cols[0]] = gridA[i];
          probe[rowIndex][cols[1]] = gridB[j];
        }
        const response = getResponse(estimator, probe, responseMethodUsed, target);
        let total = 0;
        for (let rowIndex = 0; rowIndex < response.length; rowIndex += 1) {
          total += response[rowIndex];
          ind2d[rowIndex][i][j] = response[rowIndex];
        }
        avg2d[i][j] = total / response.length;
      }
    }
    average.push(avg2d);
    individual.push(ind2d);
  }

  return {
    features: features.slice(),
    values,
    gridValues: values,
    average,
    individual: kind === "individual" || kind === "both" ? individual : undefined,
    deciles,
    responseMethodUsed,
  };
}
