import type { Matrix, Vector } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";

export type KBinsEncode = "onehot" | "onehot-dense" | "ordinal";
export type KBinsStrategy = "uniform" | "quantile" | "kmeans";

export interface KBinsDiscretizerOptions {
  nBins?: number | number[];
  encode?: KBinsEncode;
  strategy?: KBinsStrategy;
}

function quantile(sorted: Vector, q: number): number {
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

function kmeans1d(values: Vector, k: number, maxIter = 100): Vector {
  const sorted = values.slice().sort((a, b) => a - b);
  const centers = new Array<number>(k);
  for (let i = 0; i < k; i += 1) {
    const q = k === 1 ? 0.5 : i / (k - 1);
    centers[i] = quantile(sorted, q);
  }

  for (let iter = 0; iter < maxIter; iter += 1) {
    const buckets: number[][] = Array.from({ length: k }, () => []);
    for (let i = 0; i < values.length; i += 1) {
      let best = 0;
      let bestDist = Math.abs(values[i] - centers[0]);
      for (let c = 1; c < k; c += 1) {
        const d = Math.abs(values[i] - centers[c]);
        if (d < bestDist) {
          bestDist = d;
          best = c;
        }
      }
      buckets[best].push(values[i]);
    }

    let changed = 0;
    for (let c = 0; c < k; c += 1) {
      if (buckets[c].length === 0) {
        continue;
      }
      let sum = 0;
      for (let i = 0; i < buckets[c].length; i += 1) {
        sum += buckets[c][i];
      }
      const updated = sum / buckets[c].length;
      changed = Math.max(changed, Math.abs(updated - centers[c]));
      centers[c] = updated;
    }
    if (changed < 1e-6) {
      break;
    }
  }

  centers.sort((a, b) => a - b);
  return centers;
}

export class KBinsDiscretizer {
  binEdges_: Matrix | null = null;
  nBins_: Vector | null = null;
  nFeaturesIn_: number | null = null;

  private nBins: number | number[];
  private encode: KBinsEncode;
  private strategy: KBinsStrategy;
  private fitted = false;

  constructor(options: KBinsDiscretizerOptions = {}) {
    this.nBins = options.nBins ?? 5;
    this.encode = options.encode ?? "onehot";
    this.strategy = options.strategy ?? "quantile";
    if (!(this.encode === "onehot" || this.encode === "onehot-dense" || this.encode === "ordinal")) {
      throw new Error(`encode must be one of onehot/onehot-dense/ordinal. Got ${this.encode}.`);
    }
    if (!(this.strategy === "uniform" || this.strategy === "quantile" || this.strategy === "kmeans")) {
      throw new Error(`strategy must be one of uniform/quantile/kmeans. Got ${this.strategy}.`);
    }
  }

  fit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    const nFeatures = X[0].length;
    const nBins = Array.isArray(this.nBins)
      ? this.nBins.slice()
      : new Array<number>(nFeatures).fill(this.nBins);
    if (nBins.length !== nFeatures) {
      throw new Error(`nBins array must match number of features ${nFeatures}.`);
    }
    for (let i = 0; i < nBins.length; i += 1) {
      if (!Number.isInteger(nBins[i]) || nBins[i] < 2) {
        throw new Error(`nBins must contain integers >= 2. Got ${nBins[i]} at index ${i}.`);
      }
    }

    const edges: Matrix = new Array(nFeatures);
    for (let j = 0; j < nFeatures; j += 1) {
      const values = X.map((row) => row[j]);
      const sorted = values.slice().sort((a, b) => a - b);
      const bins = nBins[j];
      const featureEdges = new Array<number>(bins + 1);

      if (this.strategy === "uniform") {
        const min = sorted[0];
        const max = sorted[sorted.length - 1];
        const width = (max - min) / bins;
        for (let b = 0; b <= bins; b += 1) {
          featureEdges[b] = b === bins ? max : min + width * b;
        }
      } else if (this.strategy === "quantile") {
        for (let b = 0; b <= bins; b += 1) {
          featureEdges[b] = quantile(sorted, b / bins);
        }
      } else {
        const centers = kmeans1d(values, bins);
        featureEdges[0] = sorted[0];
        featureEdges[bins] = sorted[sorted.length - 1];
        for (let b = 1; b < bins; b += 1) {
          featureEdges[b] = 0.5 * (centers[b - 1] + centers[b]);
        }
      }

      for (let b = 1; b < featureEdges.length; b += 1) {
        if (featureEdges[b] <= featureEdges[b - 1]) {
          featureEdges[b] = featureEdges[b - 1] + 1e-12;
        }
      }
      edges[j] = featureEdges;
    }

    this.binEdges_ = edges;
    this.nBins_ = nBins;
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

    const ordinal: Matrix = new Array(X.length);
    for (let i = 0; i < X.length; i += 1) {
      const row = new Array<number>(this.nFeaturesIn_!);
      for (let j = 0; j < this.nFeaturesIn_!; j += 1) {
        const edges = this.binEdges_![j];
        let bin = edges.length - 2;
        for (let b = 1; b < edges.length; b += 1) {
          if (X[i][j] < edges[b]) {
            bin = b - 1;
            break;
          }
        }
        row[j] = bin;
      }
      ordinal[i] = row;
    }

    if (this.encode === "ordinal") {
      return ordinal;
    }

    const totalBins = this.nBins_!.reduce((sum, value) => sum + value, 0);
    const out: Matrix = new Array(X.length);
    for (let i = 0; i < ordinal.length; i += 1) {
      const row = new Array<number>(totalBins).fill(0);
      let offset = 0;
      for (let j = 0; j < this.nBins_!.length; j += 1) {
        row[offset + ordinal[i][j]] = 1;
        offset += this.nBins_![j];
      }
      out[i] = row;
    }
    return out;
  }

  fitTransform(X: Matrix): Matrix {
    return this.fit(X).transform(X);
  }

  inverseTransform(X: Matrix): Matrix {
    this.assertFitted();
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    let ordinal: Matrix;
    if (this.encode === "ordinal") {
      ordinal = X.map((row) => row.slice());
      if (ordinal[0].length !== this.nFeaturesIn_) {
        throw new Error(`Ordinal matrix must have ${this.nFeaturesIn_} columns.`);
      }
    } else {
      const expected = this.nBins_!.reduce((sum, value) => sum + value, 0);
      if (X[0].length !== expected) {
        throw new Error(`One-hot matrix must have ${expected} columns.`);
      }
      ordinal = new Array(X.length);
      for (let i = 0; i < X.length; i += 1) {
        const row = new Array<number>(this.nFeaturesIn_!);
        let offset = 0;
        for (let j = 0; j < this.nBins_!.length; j += 1) {
          let best = 0;
          let bestValue = X[i][offset];
          for (let b = 1; b < this.nBins_![j]; b += 1) {
            const value = X[i][offset + b];
            if (value > bestValue) {
              bestValue = value;
              best = b;
            }
          }
          row[j] = best;
          offset += this.nBins_![j];
        }
        ordinal[i] = row;
      }
    }

    const out: Matrix = new Array(ordinal.length);
    for (let i = 0; i < ordinal.length; i += 1) {
      const row = new Array<number>(this.nFeaturesIn_!);
      for (let j = 0; j < this.nFeaturesIn_!; j += 1) {
        const edges = this.binEdges_![j];
        const bin = Math.max(0, Math.min(edges.length - 2, Math.round(ordinal[i][j])));
        row[j] = 0.5 * (edges[bin] + edges[bin + 1]);
      }
      out[i] = row;
    }
    return out;
  }

  private assertFitted(): void {
    if (!this.fitted || !this.binEdges_ || !this.nBins_ || this.nFeaturesIn_ === null) {
      throw new Error("KBinsDiscretizer has not been fitted.");
    }
  }
}
