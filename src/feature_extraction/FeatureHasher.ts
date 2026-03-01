import type { Matrix } from "../types";

export type FeatureHasherInputType = "dict" | "pair" | "string";
export type HashedDictSample = Record<string, number | string | boolean>;
export type HashedPairSample = Array<[string, number]>;
export type HashedStringSample = string[];

export interface FeatureHasherOptions {
  nFeatures?: number;
  inputType?: FeatureHasherInputType;
  alternateSign?: boolean;
}

function fnv1aHash(input: string): number {
  let hash = 2166136261;
  for (let i = 0; i < input.length; i += 1) {
    hash ^= input.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function normalizeDictEntry(key: string, value: number | string | boolean): { token: string; amount: number } | null {
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      throw new Error(`FeatureHasher encountered non-finite numeric value for key '${key}'.`);
    }
    return { token: key, amount: value };
  }
  if (typeof value === "boolean") {
    return value ? { token: key, amount: 1 } : null;
  }
  return { token: `${key}=${value}`, amount: 1 };
}

export class FeatureHasher {
  readonly nFeatures: number;
  readonly inputType: FeatureHasherInputType;
  readonly alternateSign: boolean;

  constructor(options: FeatureHasherOptions = {}) {
    this.nFeatures = options.nFeatures ?? 1048576;
    this.inputType = options.inputType ?? "dict";
    this.alternateSign = options.alternateSign ?? true;

    if (!Number.isInteger(this.nFeatures) || this.nFeatures < 1) {
      throw new Error(`nFeatures must be an integer >= 1. Got ${this.nFeatures}.`);
    }
    if (!(this.inputType === "dict" || this.inputType === "pair" || this.inputType === "string")) {
      throw new Error(`inputType must be 'dict', 'pair', or 'string'. Got ${this.inputType}.`);
    }
  }

  fit(_X: Array<HashedDictSample | HashedPairSample | HashedStringSample>): this {
    // Stateless transformer for sklearn parity.
    return this;
  }

  transform(X: Array<HashedDictSample | HashedPairSample | HashedStringSample>): Matrix {
    if (!Array.isArray(X) || X.length === 0) {
      throw new Error("X must be a non-empty array.");
    }

    const out: Matrix = Array.from({ length: X.length }, () => new Array<number>(this.nFeatures).fill(0));
    for (let i = 0; i < X.length; i += 1) {
      if (this.inputType === "dict") {
        this.hashDictSample(out[i], X[i] as HashedDictSample);
      } else if (this.inputType === "pair") {
        this.hashPairSample(out[i], X[i] as HashedPairSample);
      } else {
        this.hashStringSample(out[i], X[i] as HashedStringSample);
      }
    }
    return out;
  }

  fitTransform(X: Array<HashedDictSample | HashedPairSample | HashedStringSample>): Matrix {
    return this.fit(X).transform(X);
  }

  private hashDictSample(target: number[], sample: HashedDictSample): void {
    const keys = Object.keys(sample);
    for (let i = 0; i < keys.length; i += 1) {
      const key = keys[i];
      const normalized = normalizeDictEntry(key, sample[key]);
      if (!normalized) {
        continue;
      }
      this.addHashedValue(target, normalized.token, normalized.amount);
    }
  }

  private hashPairSample(target: number[], sample: HashedPairSample): void {
    for (let i = 0; i < sample.length; i += 1) {
      const [token, amount] = sample[i];
      if (!Number.isFinite(amount)) {
        throw new Error(`FeatureHasher pair value must be finite. Got ${amount}.`);
      }
      this.addHashedValue(target, token, amount);
    }
  }

  private hashStringSample(target: number[], sample: HashedStringSample): void {
    for (let i = 0; i < sample.length; i += 1) {
      this.addHashedValue(target, sample[i], 1);
    }
  }

  private addHashedValue(target: number[], token: string, amount: number): void {
    const hash = fnv1aHash(token);
    const index = hash % this.nFeatures;
    const sign = this.alternateSign && ((hash >>> 31) & 1) === 1 ? -1 : 1;
    target[index] += sign * amount;
  }
}

