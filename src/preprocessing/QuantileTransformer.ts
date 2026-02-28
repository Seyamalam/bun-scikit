import type { Matrix, Vector } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";

export type QuantileOutputDistribution = "uniform" | "normal";

export interface QuantileTransformerOptions {
  nQuantiles?: number;
  outputDistribution?: QuantileOutputDistribution;
}

function erf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;
  const t = 1 / (1 + p * ax);
  const y = 1 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * Math.exp(-ax * ax);
  return sign * y;
}

function normalCdf(x: number): number {
  return 0.5 * (1 + erf(x / Math.sqrt(2)));
}

// Acklam approximation for inverse normal CDF.
function inverseNormalCdf(p: number): number {
  const pp = Math.min(1 - 1e-12, Math.max(1e-12, p));
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2, -3.066479806614716e1, 2.506628277459239];
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
  const plow = 0.02425;
  const phigh = 1 - plow;

  if (pp < plow) {
    const q = Math.sqrt(-2 * Math.log(pp));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (pp > phigh) {
    const q = Math.sqrt(-2 * Math.log(1 - pp));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  const q = pp - 0.5;
  const r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
    (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

function quantileAt(sorted: Vector, q: number): number {
  if (sorted.length === 1) {
    return sorted[0];
  }
  const pos = q * (sorted.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) {
    return sorted[lo];
  }
  const alpha = pos - lo;
  return sorted[lo] * (1 - alpha) + sorted[hi] * alpha;
}

function searchSorted(sorted: Vector, value: number): number {
  let lo = 0;
  let hi = sorted.length - 1;
  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (sorted[mid] < value) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

export class QuantileTransformer {
  quantiles_: Matrix | null = null;
  references_: Vector | null = null;
  nQuantiles_: number | null = null;
  nFeaturesIn_: number | null = null;

  private nQuantiles: number;
  private outputDistribution: QuantileOutputDistribution;
  private quantilesByFeature: Matrix | null = null;
  private fitted = false;

  constructor(options: QuantileTransformerOptions = {}) {
    this.nQuantiles = options.nQuantiles ?? 1000;
    this.outputDistribution = options.outputDistribution ?? "uniform";
    if (!Number.isInteger(this.nQuantiles) || this.nQuantiles < 1) {
      throw new Error(`nQuantiles must be an integer >= 1. Got ${this.nQuantiles}.`);
    }
    if (!(this.outputDistribution === "uniform" || this.outputDistribution === "normal")) {
      throw new Error(
        `outputDistribution must be 'uniform' or 'normal'. Got ${this.outputDistribution}.`,
      );
    }
  }

  fit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    const nSamples = X.length;
    const nFeatures = X[0].length;
    const nQuantiles = Math.min(this.nQuantiles, nSamples);
    const references = new Array<number>(nQuantiles);
    for (let i = 0; i < nQuantiles; i += 1) {
      references[i] = nQuantiles === 1 ? 0 : i / (nQuantiles - 1);
    }

    const quantilesByFeature: Matrix = new Array(nFeatures);
    for (let j = 0; j < nFeatures; j += 1) {
      const sorted = X.map((row) => row[j]).sort((a, b) => a - b);
      const featureQuantiles = new Array<number>(nQuantiles);
      for (let i = 0; i < nQuantiles; i += 1) {
        featureQuantiles[i] = quantileAt(sorted, references[i]);
      }
      quantilesByFeature[j] = featureQuantiles;
    }

    const quantiles: Matrix = Array.from({ length: nQuantiles }, () =>
      new Array<number>(nFeatures).fill(0),
    );
    for (let i = 0; i < nQuantiles; i += 1) {
      for (let j = 0; j < nFeatures; j += 1) {
        quantiles[i][j] = quantilesByFeature[j][i];
      }
    }

    this.quantiles_ = quantiles;
    this.quantilesByFeature = quantilesByFeature;
    this.references_ = references;
    this.nQuantiles_ = nQuantiles;
    this.nFeaturesIn_ = nFeatures;
    this.fitted = true;
    return this;
  }

  transform(X: Matrix): Matrix {
    this.assertFitted();
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }

    const out: Matrix = new Array(X.length);
    for (let i = 0; i < X.length; i += 1) {
      const row = new Array<number>(this.nFeaturesIn_!);
      for (let j = 0; j < this.nFeaturesIn_!; j += 1) {
        const quantiles = this.quantilesByFeature![j];
        const refs = this.references_!;
        if (X[i][j] <= quantiles[0]) {
          row[j] = refs[0];
        } else if (X[i][j] >= quantiles[quantiles.length - 1]) {
          row[j] = refs[refs.length - 1];
        } else {
          const idx = searchSorted(quantiles, X[i][j]);
          const left = Math.max(0, idx - 1);
          const right = idx;
          const qLeft = quantiles[left];
          const qRight = quantiles[right];
          const rLeft = refs[left];
          const rRight = refs[right];
          const ratio = qRight === qLeft ? 0 : (X[i][j] - qLeft) / (qRight - qLeft);
          row[j] = rLeft * (1 - ratio) + rRight * ratio;
        }
        if (this.outputDistribution === "normal") {
          row[j] = inverseNormalCdf(row[j]);
        }
      }
      out[i] = row;
    }
    return out;
  }

  inverseTransform(X: Matrix): Matrix {
    this.assertFitted();
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }

    const out: Matrix = new Array(X.length);
    for (let i = 0; i < X.length; i += 1) {
      const row = new Array<number>(this.nFeaturesIn_!);
      for (let j = 0; j < this.nFeaturesIn_!; j += 1) {
        const quantiles = this.quantilesByFeature![j];
        const refs = this.references_!;
        const target =
          this.outputDistribution === "normal"
            ? normalCdf(X[i][j])
            : X[i][j];
        const clipped = Math.min(1, Math.max(0, target));

        if (clipped <= refs[0]) {
          row[j] = quantiles[0];
        } else if (clipped >= refs[refs.length - 1]) {
          row[j] = quantiles[quantiles.length - 1];
        } else {
          const idx = searchSorted(refs, clipped);
          const left = Math.max(0, idx - 1);
          const right = idx;
          const rLeft = refs[left];
          const rRight = refs[right];
          const qLeft = quantiles[left];
          const qRight = quantiles[right];
          const ratio = rRight === rLeft ? 0 : (clipped - rLeft) / (rRight - rLeft);
          row[j] = qLeft * (1 - ratio) + qRight * ratio;
        }
      }
      out[i] = row;
    }
    return out;
  }

  fitTransform(X: Matrix): Matrix {
    return this.fit(X).transform(X);
  }

  private assertFitted(): void {
    if (
      !this.fitted ||
      !this.quantiles_ ||
      !this.references_ ||
      !this.quantilesByFeature ||
      this.nQuantiles_ === null ||
      this.nFeaturesIn_ === null
    ) {
      throw new Error("QuantileTransformer has not been fitted.");
    }
  }
}
