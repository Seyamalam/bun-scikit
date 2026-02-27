import type { Matrix, Vector } from "../types";
import { r2Score } from "../metrics/regression";
import { assertFiniteVector, validateRegressionInputs } from "../utils/validation";

export type DummyRegressorStrategy = "mean" | "median" | "quantile" | "constant";

export interface DummyRegressorOptions {
  strategy?: DummyRegressorStrategy;
  constant?: number;
  quantile?: number;
}

function computeMedian(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) {
    return 0.5 * (sorted[mid - 1] + sorted[mid]);
  }
  return sorted[mid];
}

function computeQuantile(values: number[], q: number): number {
  const sorted = [...values].sort((a, b) => a - b);
  const pos = q * (sorted.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) {
    return sorted[lo];
  }
  const weight = pos - lo;
  return sorted[lo] * (1 - weight) + sorted[hi] * weight;
}

export class DummyRegressor {
  constant_: number | null = null;

  private readonly strategy: DummyRegressorStrategy;
  private readonly constant?: number;
  private readonly quantile: number;
  private nFeaturesIn_: number | null = null;

  constructor(options: DummyRegressorOptions = {}) {
    this.strategy = options.strategy ?? "mean";
    this.constant = options.constant;
    this.quantile = options.quantile ?? 0.5;

    if (this.strategy === "constant") {
      if (!Number.isFinite(this.constant)) {
        throw new Error("constant strategy requires a finite constant value.");
      }
    }

    if (this.strategy === "quantile") {
      if (!Number.isFinite(this.quantile) || this.quantile < 0 || this.quantile > 1) {
        throw new Error(`quantile must be in [0, 1]. Got ${this.quantile}.`);
      }
    }
  }

  fit(X: Matrix, y: Vector, sampleWeight?: Vector): this {
    validateRegressionInputs(X, y);
    this.nFeaturesIn_ = X[0].length;

    switch (this.strategy) {
      case "mean": {
        let total = 0;
        for (let i = 0; i < y.length; i += 1) {
          total += y[i];
        }
        this.constant_ = total / y.length;
        break;
      }
      case "median":
        this.constant_ = computeMedian(y);
        break;
      case "quantile":
        this.constant_ = computeQuantile(y, this.quantile);
        break;
      case "constant":
        this.constant_ = this.constant!;
        break;
      default: {
        const exhaustive: never = this.strategy;
        throw new Error(`Unsupported strategy: ${exhaustive}`);
      }
    }

    return this;
  }

  predict(X: Matrix): Vector {
    if (this.constant_ === null || this.nFeaturesIn_ === null) {
      throw new Error("DummyRegressor has not been fitted.");
    }
    if (!Array.isArray(X) || X.length === 0) {
      throw new Error("X must be a non-empty 2D array.");
    }
    if (!Array.isArray(X[0]) || X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0]?.length ?? 0}.`);
    }
    return new Array<number>(X.length).fill(this.constant_);
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return r2Score(y, this.predict(X));
  }
}

