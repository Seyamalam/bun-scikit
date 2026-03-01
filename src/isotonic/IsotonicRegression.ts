import type { RegressionModel, Vector } from "../types";
import { r2Score } from "../metrics/regression";
import { assertFiniteVector } from "../utils/validation";

export type IsotonicIncreasing = boolean | "auto";
export type IsotonicOutOfBounds = "nan" | "clip" | "raise";

export interface IsotonicRegressionOptions {
  yMin?: number;
  yMax?: number;
  increasing?: IsotonicIncreasing;
  outOfBounds?: IsotonicOutOfBounds;
}

interface Block {
  start: number;
  end: number;
  weight: number;
  value: number;
}

function argsort(values: Vector): number[] {
  return Array.from({ length: values.length }, (_, i) => i).sort((a, b) => values[a] - values[b]);
}

function pearsonCorrelation(a: Vector, b: Vector): number {
  const n = a.length;
  let meanA = 0;
  let meanB = 0;
  for (let i = 0; i < n; i += 1) {
    meanA += a[i];
    meanB += b[i];
  }
  meanA /= n;
  meanB /= n;

  let num = 0;
  let denA = 0;
  let denB = 0;
  for (let i = 0; i < n; i += 1) {
    const da = a[i] - meanA;
    const db = b[i] - meanB;
    num += da * db;
    denA += da * da;
    denB += db * db;
  }
  if (denA <= 1e-12 || denB <= 1e-12) {
    return 0;
  }
  return num / Math.sqrt(denA * denB);
}

export class IsotonicRegression implements RegressionModel {
  XThresholds_: Vector | null = null;
  yThresholds_: Vector | null = null;
  increasing_: boolean | null = null;

  private yMin?: number;
  private yMax?: number;
  private increasing: IsotonicIncreasing;
  private outOfBounds: IsotonicOutOfBounds;
  private fitted = false;

  constructor(options: IsotonicRegressionOptions = {}) {
    this.yMin = options.yMin;
    this.yMax = options.yMax;
    this.increasing = options.increasing ?? true;
    this.outOfBounds = options.outOfBounds ?? "nan";

    if (this.yMin !== undefined && !Number.isFinite(this.yMin)) {
      throw new Error(`yMin must be finite. Got ${this.yMin}.`);
    }
    if (this.yMax !== undefined && !Number.isFinite(this.yMax)) {
      throw new Error(`yMax must be finite. Got ${this.yMax}.`);
    }
    if (
      this.yMin !== undefined &&
      this.yMax !== undefined &&
      this.yMin > this.yMax
    ) {
      throw new Error(`yMin must be <= yMax. Got yMin=${this.yMin}, yMax=${this.yMax}.`);
    }
    if (!(this.outOfBounds === "nan" || this.outOfBounds === "clip" || this.outOfBounds === "raise")) {
      throw new Error(`outOfBounds must be 'nan', 'clip', or 'raise'. Got ${this.outOfBounds}.`);
    }
  }

  fit(X: Vector[], y: Vector, sampleWeight?: Vector): this {
    if (!Array.isArray(X) || X.length === 0) {
      throw new Error("X must be a non-empty 2D array with one feature.");
    }
    if (X[0].length !== 1) {
      throw new Error("IsotonicRegression expects exactly one feature column.");
    }
    if (y.length !== X.length) {
      throw new Error(`X and y length mismatch. Got ${X.length} and ${y.length}.`);
    }
    assertFiniteVector(y);

    const x = X.map((row) => row[0]);
    assertFiniteVector(x, "X[:, 0]");

    let weights = sampleWeight?.slice() ?? new Array<number>(x.length).fill(1);
    assertFiniteVector(weights, "sampleWeight");
    for (let i = 0; i < weights.length; i += 1) {
      if (weights[i] <= 0) {
        throw new Error(`sampleWeight values must be > 0. Got ${weights[i]} at index ${i}.`);
      }
    }

    const order = argsort(x);
    const xSorted = new Array<number>(x.length);
    const ySorted = new Array<number>(y.length);
    const wSorted = new Array<number>(weights.length);
    for (let i = 0; i < order.length; i += 1) {
      xSorted[i] = x[order[i]];
      ySorted[i] = y[order[i]];
      wSorted[i] = weights[order[i]];
    }

    const increasing = this.increasing === "auto" ? pearsonCorrelation(xSorted, ySorted) >= 0 : this.increasing;
    this.increasing_ = increasing;

    const blocks: Block[] = [];
    for (let i = 0; i < ySorted.length; i += 1) {
      blocks.push({ start: i, end: i, weight: wSorted[i], value: ySorted[i] });
      while (blocks.length >= 2) {
        const b = blocks[blocks.length - 1];
        const a = blocks[blocks.length - 2];
        const violation = increasing ? a.value > b.value : a.value < b.value;
        if (!violation) {
          break;
        }
        const mergedWeight = a.weight + b.weight;
        const mergedValue = (a.value * a.weight + b.value * b.weight) / mergedWeight;
        blocks.splice(blocks.length - 2, 2, {
          start: a.start,
          end: b.end,
          weight: mergedWeight,
          value: mergedValue,
        });
      }
    }

    const yIsoSorted = new Array<number>(ySorted.length);
    for (let i = 0; i < blocks.length; i += 1) {
      let value = blocks[i].value;
      if (this.yMin !== undefined) {
        value = Math.max(this.yMin, value);
      }
      if (this.yMax !== undefined) {
        value = Math.min(this.yMax, value);
      }
      for (let j = blocks[i].start; j <= blocks[i].end; j += 1) {
        yIsoSorted[j] = value;
      }
    }

    this.XThresholds_ = xSorted;
    this.yThresholds_ = yIsoSorted;
    this.fitted = true;
    return this;
  }

  predict(X: Vector[]): Vector {
    this.assertFitted();
    if (!Array.isArray(X) || X.length === 0) {
      throw new Error("X must be a non-empty 2D array with one feature.");
    }
    if (X[0].length !== 1) {
      throw new Error("IsotonicRegression expects exactly one feature column.");
    }
    const x = X.map((row) => row[0]);
    assertFiniteVector(x, "X[:, 0]");

    const out = new Array<number>(x.length).fill(0);
    const xs = this.XThresholds_!;
    const ys = this.yThresholds_!;
    const left = xs[0];
    const right = xs[xs.length - 1];

    for (let i = 0; i < x.length; i += 1) {
      const value = x[i];
      if (value < left) {
        if (this.outOfBounds === "clip") {
          out[i] = ys[0];
          continue;
        }
        if (this.outOfBounds === "raise") {
          throw new Error(`Value ${value} is below fitted domain minimum ${left}.`);
        }
        out[i] = Number.NaN;
        continue;
      }
      if (value > right) {
        if (this.outOfBounds === "clip") {
          out[i] = ys[ys.length - 1];
          continue;
        }
        if (this.outOfBounds === "raise") {
          throw new Error(`Value ${value} is above fitted domain maximum ${right}.`);
        }
        out[i] = Number.NaN;
        continue;
      }

      let hi = 0;
      while (hi < xs.length && xs[hi] < value) {
        hi += 1;
      }
      if (hi === 0) {
        out[i] = ys[0];
      } else if (hi >= xs.length) {
        out[i] = ys[ys.length - 1];
      } else {
        const lo = hi - 1;
        const x0 = xs[lo];
        const x1 = xs[hi];
        const y0 = ys[lo];
        const y1 = ys[hi];
        if (x1 === x0) {
          out[i] = y1;
        } else {
          const t = (value - x0) / (x1 - x0);
          out[i] = y0 * (1 - t) + y1 * t;
        }
      }
    }
    return out;
  }

  transform(X: Vector[]): Vector {
    return this.predict(X);
  }

  score(X: Vector[], y: Vector): number {
    assertFiniteVector(y);
    return r2Score(y, this.predict(X));
  }

  private assertFitted(): void {
    if (!this.fitted || this.XThresholds_ === null || this.yThresholds_ === null || this.increasing_ === null) {
      throw new Error("IsotonicRegression has not been fitted.");
    }
  }
}

