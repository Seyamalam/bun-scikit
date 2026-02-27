import type { Matrix, Vector } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";

export type LocalOutlierFactorContamination = "auto" | number;

export interface LocalOutlierFactorOptions {
  nNeighbors?: number;
  contamination?: LocalOutlierFactorContamination;
  novelty?: boolean;
}

function euclideanDistance(a: Vector, b: Vector): number {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

function quantile(values: Vector, q: number): number {
  if (values.length === 0) {
    return 0;
  }
  const sorted = values.slice().sort((a, b) => a - b);
  const pos = Math.max(0, Math.min(sorted.length - 1, q * (sorted.length - 1)));
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) {
    return sorted[lo];
  }
  const alpha = pos - lo;
  return sorted[lo] * (1 - alpha) + sorted[hi] * alpha;
}

function neighborIndices(X: Matrix, sample: Vector, k: number, skipSelfIndex?: number): number[] {
  const pairs: Array<{ index: number; distance: number }> = [];
  for (let i = 0; i < X.length; i += 1) {
    if (skipSelfIndex !== undefined && i === skipSelfIndex) {
      continue;
    }
    pairs.push({ index: i, distance: euclideanDistance(sample, X[i]) });
  }
  pairs.sort((a, b) => a.distance - b.distance);
  return pairs.slice(0, Math.min(k, pairs.length)).map((p) => p.index);
}

function localReachabilityDensities(X: Matrix, nNeighbors: number): Vector {
  const neighborIdx = X.map((row, idx) => neighborIndices(X, row, nNeighbors, idx));
  const kDistance = neighborIdx.map((indices, idx) => {
    if (indices.length === 0) {
      return 0;
    }
    return euclideanDistance(X[idx], X[indices[indices.length - 1]]);
  });

  const lrd = new Array<number>(X.length).fill(0);
  for (let i = 0; i < X.length; i += 1) {
    const neighbors = neighborIdx[i];
    if (neighbors.length === 0) {
      lrd[i] = 0;
      continue;
    }
    let reachDistSum = 0;
    for (let j = 0; j < neighbors.length; j += 1) {
      const n = neighbors[j];
      const dist = euclideanDistance(X[i], X[n]);
      reachDistSum += Math.max(kDistance[n], dist);
    }
    lrd[i] = neighbors.length / Math.max(reachDistSum, 1e-12);
  }
  return lrd;
}

function lofScores(X: Matrix, nNeighbors: number, baseLrd?: Vector): Vector {
  const neighborIdx = X.map((row, idx) => neighborIndices(X, row, nNeighbors, idx));
  const lrd = baseLrd ?? localReachabilityDensities(X, nNeighbors);
  const scores = new Array<number>(X.length).fill(1);
  for (let i = 0; i < X.length; i += 1) {
    const neighbors = neighborIdx[i];
    if (neighbors.length === 0 || lrd[i] <= 1e-12) {
      scores[i] = 1;
      continue;
    }
    let ratioSum = 0;
    for (let j = 0; j < neighbors.length; j += 1) {
      ratioSum += lrd[neighbors[j]] / lrd[i];
    }
    scores[i] = ratioSum / neighbors.length;
  }
  return scores;
}

export class LocalOutlierFactor {
  negativeOutlierFactor_: Vector | null = null;
  offset_ = -1.5;
  nFeaturesIn_: number | null = null;

  private nNeighbors: number;
  private contamination: LocalOutlierFactorContamination;
  private novelty: boolean;
  private XTrain: Matrix | null = null;
  private lrdTrain: Vector | null = null;
  private fitted = false;

  constructor(options: LocalOutlierFactorOptions = {}) {
    this.nNeighbors = options.nNeighbors ?? 20;
    this.contamination = options.contamination ?? "auto";
    this.novelty = options.novelty ?? false;

    if (!Number.isInteger(this.nNeighbors) || this.nNeighbors < 1) {
      throw new Error(`nNeighbors must be an integer >= 1. Got ${this.nNeighbors}.`);
    }
    if (
      this.contamination !== "auto" &&
      (!Number.isFinite(this.contamination) || this.contamination <= 0 || this.contamination >= 0.5)
    ) {
      throw new Error(
        `contamination must be 'auto' or a finite value in (0, 0.5). Got ${this.contamination}.`,
      );
    }
  }

  fit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    const k = Math.min(this.nNeighbors, Math.max(1, X.length - 1));
    this.XTrain = X.map((row) => row.slice());
    this.nFeaturesIn_ = X[0].length;
    this.lrdTrain = localReachabilityDensities(this.XTrain, k);
    const lof = lofScores(this.XTrain, k, this.lrdTrain);
    this.negativeOutlierFactor_ = lof.map((value) => -value);
    const contamination = this.contamination === "auto" ? 0.1 : this.contamination;
    this.offset_ = quantile(this.negativeOutlierFactor_, contamination);
    this.fitted = true;
    return this;
  }

  fitPredict(X: Matrix): Vector {
    this.fit(X);
    return this.negativeOutlierFactor_!.map((value) => (value >= this.offset_ ? 1 : -1));
  }

  scoreSamples(X: Matrix): Vector {
    this.assertFitted();
    if (!this.novelty) {
      throw new Error("scoreSamples is only available when novelty=true.");
    }
    this.validatePredictInput(X);
    const k = Math.min(this.nNeighbors, Math.max(1, this.XTrain!.length - 1));
    const scores = new Array<number>(X.length).fill(-1);
    for (let i = 0; i < X.length; i += 1) {
      const neighbors = neighborIndices(this.XTrain!, X[i], k);
      if (neighbors.length === 0) {
        scores[i] = -1;
        continue;
      }
      let reachDistSum = 0;
      for (let j = 0; j < neighbors.length; j += 1) {
        const n = neighbors[j];
        const nNeighbors = neighborIndices(this.XTrain!, this.XTrain![n], k, n);
        const kDistance = nNeighbors.length === 0
          ? 0
          : euclideanDistance(this.XTrain![n], this.XTrain![nNeighbors[nNeighbors.length - 1]]);
        const dist = euclideanDistance(X[i], this.XTrain![n]);
        reachDistSum += Math.max(kDistance, dist);
      }
      const lrdQuery = neighbors.length / Math.max(reachDistSum, 1e-12);
      let lof = 0;
      for (let j = 0; j < neighbors.length; j += 1) {
        lof += (this.lrdTrain![neighbors[j]] ?? 0) / Math.max(lrdQuery, 1e-12);
      }
      lof /= neighbors.length;
      scores[i] = -lof;
    }
    return scores;
  }

  decisionFunction(X: Matrix): Vector {
    return this.scoreSamples(X).map((value) => value - this.offset_);
  }

  predict(X: Matrix): Vector {
    return this.decisionFunction(X).map((value) => (value >= 0 ? 1 : -1));
  }

  private validatePredictInput(X: Matrix): void {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }
  }

  private assertFitted(): void {
    if (!this.fitted || !this.XTrain || !this.lrdTrain || this.nFeaturesIn_ === null) {
      throw new Error("LocalOutlierFactor has not been fitted.");
    }
  }
}
