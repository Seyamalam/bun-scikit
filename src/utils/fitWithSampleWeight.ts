import type { Matrix, Vector } from "../types";

export interface FitSampleWeightRequest {
  sampleWeight?: boolean;
}

export interface FitCapable {
  fit(X: Matrix, y?: Vector, sampleWeight?: Vector): unknown;
}

function resolveSampleWeightRouting(
  estimator: unknown,
  request: FitSampleWeightRequest | undefined,
): boolean {
  if (typeof request?.sampleWeight === "boolean") {
    return request.sampleWeight;
  }
  if (typeof estimator !== "object" || estimator === null) {
    return true;
  }
  const marker = (estimator as Record<string, unknown>).sampleWeightRequest_;
  if (typeof marker === "boolean") {
    return marker;
  }
  return true;
}

export function fitWithSampleWeight(
  estimator: FitCapable,
  X: Matrix,
  y?: Vector,
  sampleWeight?: Vector,
  request?: FitSampleWeightRequest,
): void {
  if (sampleWeight && resolveSampleWeightRouting(estimator, request)) {
    estimator.fit(X, y, sampleWeight);
    return;
  }
  estimator.fit(X, y);
}

export function subsetSampleWeight(
  sampleWeight: Vector | undefined,
  indices: number[],
): Vector | undefined {
  if (!sampleWeight) {
    return undefined;
  }
  const out = new Array<number>(indices.length);
  for (let i = 0; i < indices.length; i += 1) {
    out[i] = sampleWeight[indices[i]];
  }
  return out;
}
