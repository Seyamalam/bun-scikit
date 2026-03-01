import type { Matrix, Vector } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";
import { MinCovDet } from "./MinCovDet";

export interface EllipticEnvelopeOptions {
  contamination?: number;
  supportFraction?: number;
  maxIter?: number;
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
  const t = pos - lo;
  return sorted[lo] * (1 - t) + sorted[hi] * t;
}

export class EllipticEnvelope {
  location_: Vector | null = null;
  covariance_: Matrix | null = null;
  precision_: Matrix | null = null;
  support_: boolean[] | null = null;
  offset_: number | null = null;
  dist_: Vector | null = null;
  nFeaturesIn_: number | null = null;

  private contamination: number;
  private supportFraction: number;
  private maxIter: number;
  private estimator: MinCovDet | null = null;
  private fitted = false;

  constructor(options: EllipticEnvelopeOptions = {}) {
    this.contamination = options.contamination ?? 0.1;
    this.supportFraction = options.supportFraction ?? 0.75;
    this.maxIter = options.maxIter ?? 5;

    if (!Number.isFinite(this.contamination) || this.contamination <= 0 || this.contamination >= 0.5) {
      throw new Error(`contamination must be in (0, 0.5). Got ${this.contamination}.`);
    }
    if (!Number.isFinite(this.supportFraction) || this.supportFraction <= 0 || this.supportFraction > 1) {
      throw new Error(`supportFraction must be in (0, 1]. Got ${this.supportFraction}.`);
    }
    if (!Number.isInteger(this.maxIter) || this.maxIter < 1) {
      throw new Error(`maxIter must be an integer >= 1. Got ${this.maxIter}.`);
    }
  }

  fit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    const estimator = new MinCovDet({
      supportFraction: this.supportFraction,
      maxIter: this.maxIter,
    }).fit(X);

    const distances = estimator.mahalanobis(X);
    const squared = distances.map((value) => value * value);
    const threshold = quantile(squared, 1 - this.contamination);

    this.estimator = estimator;
    this.location_ = estimator.location_;
    this.covariance_ = estimator.covariance_;
    this.precision_ = estimator.precision_;
    this.support_ = estimator.support_;
    this.dist_ = squared;
    this.offset_ = threshold;
    this.nFeaturesIn_ = X[0].length;
    this.fitted = true;
    return this;
  }

  scoreSamples(X: Matrix): Vector {
    this.assertFitted();
    const distances = this.estimator!.mahalanobis(X);
    return distances.map((value) => -(value * value));
  }

  decisionFunction(X: Matrix): Vector {
    this.assertFitted();
    const scores = this.scoreSamples(X);
    return scores.map((value) => value + this.offset_!);
  }

  predict(X: Matrix): Vector {
    const decision = this.decisionFunction(X);
    return decision.map((value) => (value >= 0 ? 1 : -1));
  }

  score(X: Matrix): number {
    const decision = this.decisionFunction(X);
    let sum = 0;
    for (let i = 0; i < decision.length; i += 1) {
      sum += decision[i];
    }
    return sum / decision.length;
  }

  private assertFitted(): void {
    if (
      !this.fitted ||
      this.estimator === null ||
      this.location_ === null ||
      this.covariance_ === null ||
      this.precision_ === null ||
      this.support_ === null ||
      this.offset_ === null ||
      this.dist_ === null ||
      this.nFeaturesIn_ === null
    ) {
      throw new Error("EllipticEnvelope has not been fitted.");
    }
  }
}

