import type { Matrix } from "../types";

export type DictValue = number | string | boolean;
export type DictSample = Record<string, DictValue>;

export interface DictVectorizerOptions {
  sort?: boolean;
  separator?: string;
}

function tokenFromEntry(key: string, value: DictValue, separator: string): { feature: string; amount: number } | null {
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      throw new Error(`DictVectorizer encountered a non-finite numeric value for key '${key}'.`);
    }
    return { feature: key, amount: value };
  }
  if (typeof value === "boolean") {
    return value ? { feature: key, amount: 1 } : null;
  }
  return { feature: `${key}${separator}${value}`, amount: 1 };
}

export class DictVectorizer {
  vocabulary_: Record<string, number> | null = null;
  featureNames_: string[] | null = null;

  private sort: boolean;
  private separator: string;
  private fitted = false;

  constructor(options: DictVectorizerOptions = {}) {
    this.sort = options.sort ?? true;
    this.separator = options.separator ?? "=";
  }

  fit(X: DictSample[]): this {
    if (!Array.isArray(X) || X.length === 0) {
      throw new Error("X must be a non-empty array of dictionaries.");
    }

    const features = new Set<string>();
    for (let i = 0; i < X.length; i += 1) {
      const sample = X[i];
      const keys = Object.keys(sample);
      for (let k = 0; k < keys.length; k += 1) {
        const key = keys[k];
        const token = tokenFromEntry(key, sample[key], this.separator);
        if (token) {
          features.add(token.feature);
        }
      }
    }

    const featureNames = Array.from(features);
    if (this.sort) {
      featureNames.sort((a, b) => a.localeCompare(b));
    }
    const vocabulary: Record<string, number> = {};
    for (let i = 0; i < featureNames.length; i += 1) {
      vocabulary[featureNames[i]] = i;
    }

    this.vocabulary_ = vocabulary;
    this.featureNames_ = featureNames;
    this.fitted = true;
    return this;
  }

  transform(X: DictSample[]): Matrix {
    this.assertFitted();
    if (!Array.isArray(X) || X.length === 0) {
      throw new Error("X must be a non-empty array of dictionaries.");
    }

    const nFeatures = this.featureNames_!.length;
    const out: Matrix = Array.from({ length: X.length }, () => new Array<number>(nFeatures).fill(0));
    for (let i = 0; i < X.length; i += 1) {
      const sample = X[i];
      const keys = Object.keys(sample);
      for (let k = 0; k < keys.length; k += 1) {
        const key = keys[k];
        const token = tokenFromEntry(key, sample[key], this.separator);
        if (!token) {
          continue;
        }
        const index = this.vocabulary_![token.feature];
        if (index !== undefined) {
          out[i][index] += token.amount;
        }
      }
    }
    return out;
  }

  fitTransform(X: DictSample[]): Matrix {
    return this.fit(X).transform(X);
  }

  inverseTransform(X: Matrix): DictSample[] {
    this.assertFitted();
    if (!Array.isArray(X) || X.length === 0) {
      throw new Error("X must be a non-empty matrix.");
    }
    const nFeatures = this.featureNames_!.length;
    for (let i = 0; i < X.length; i += 1) {
      if (X[i].length !== nFeatures) {
        throw new Error(`Feature size mismatch. Expected ${nFeatures}, got ${X[i].length}.`);
      }
    }

    const out: DictSample[] = new Array(X.length);
    for (let i = 0; i < X.length; i += 1) {
      const sample: DictSample = {};
      for (let j = 0; j < nFeatures; j += 1) {
        const value = X[i][j];
        if (value === 0) {
          continue;
        }
        sample[this.featureNames_![j]] = value;
      }
      out[i] = sample;
    }
    return out;
  }

  getFeatureNamesOut(): string[] {
    this.assertFitted();
    return this.featureNames_!.slice();
  }

  private assertFitted(): void {
    if (!this.fitted || !this.vocabulary_ || !this.featureNames_) {
      throw new Error("DictVectorizer has not been fitted.");
    }
  }
}

