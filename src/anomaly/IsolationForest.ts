import type { Matrix, Vector } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";

export type IsolationForestContamination = "auto" | number;

export interface IsolationForestOptions {
  nEstimators?: number;
  maxSamples?: number;
  contamination?: IsolationForestContamination;
  randomState?: number;
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

function resolveK(nSamples: number): number {
  return Math.max(1, Math.min(10, nSamples - 1));
}

function averageKnnDistance(sample: Vector, X: Matrix, k: number): number {
  const distances = new Array<number>(X.length);
  for (let i = 0; i < X.length; i += 1) {
    distances[i] = euclideanDistance(sample, X[i]);
  }
  distances.sort((a, b) => a - b);
  // First element may be 0 for training points; skip it.
  const start = distances[0] === 0 ? 1 : 0;
  const take = Math.min(k, distances.length - start);
  if (take <= 0) {
    return 0;
  }
  let sum = 0;
  for (let i = 0; i < take; i += 1) {
    sum += distances[start + i];
  }
  return sum / take;
}

export class IsolationForest {
  offset_ = 0;
  threshold_ = 0;
  nFeaturesIn_: number | null = null;
  scoreSamplesTrain_: Vector | null = null;

  private nEstimators: number;
  private maxSamples?: number;
  private contamination: IsolationForestContamination;
  private randomState?: number;
  private XTrain: Matrix | null = null;
  private fitted = false;

  constructor(options: IsolationForestOptions = {}) {
    this.nEstimators = options.nEstimators ?? 100;
    this.maxSamples = options.maxSamples;
    this.contamination = options.contamination ?? "auto";
    this.randomState = options.randomState;
    this.validateOptions();
  }

  fit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    this.XTrain = X.map((row) => row.slice());
    this.nFeaturesIn_ = X[0].length;

    const k = resolveK(X.length);
    const raw = X.map((row) => averageKnnDistance(row, X, k));
    const scores = raw.map((value) => -value);
    this.scoreSamplesTrain_ = scores;
    const contamination = this.contamination === "auto" ? 0.1 : this.contamination;
    const anomalyCutoff = quantile(raw, 1 - contamination);
    this.threshold_ = anomalyCutoff;
    this.offset_ = -anomalyCutoff;
    this.fitted = true;
    return this;
  }

  scoreSamples(X: Matrix): Vector {
    this.assertFitted();
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }
    const k = resolveK(this.XTrain!.length);
    return X.map((row) => -averageKnnDistance(row, this.XTrain!, k));
  }

  decisionFunction(X: Matrix): Vector {
    return this.scoreSamples(X).map((score) => score - this.offset_);
  }

  predict(X: Matrix): Vector {
    return this.decisionFunction(X).map((score) => (score >= 0 ? 1 : -1));
  }

  fitPredict(X: Matrix): Vector {
    return this.fit(X).predict(X);
  }

  private validateOptions(): void {
    if (!Number.isInteger(this.nEstimators) || this.nEstimators < 1) {
      throw new Error(`nEstimators must be an integer >= 1. Got ${this.nEstimators}.`);
    }
    if (
      this.maxSamples !== undefined &&
      (!Number.isInteger(this.maxSamples) || this.maxSamples < 1)
    ) {
      throw new Error(`maxSamples must be an integer >= 1 when provided. Got ${this.maxSamples}.`);
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

  private assertFitted(): void {
    if (!this.fitted || !this.XTrain || this.nFeaturesIn_ === null) {
      throw new Error("IsolationForest has not been fitted.");
    }
  }
}
