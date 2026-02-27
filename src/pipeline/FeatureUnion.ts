import type { Matrix, Vector } from "../types";
import { assertConsistentRowSize, assertFiniteMatrix, assertNonEmptyMatrix } from "../utils/validation";

interface TransformerLike {
  fit(X: Matrix, y?: Vector, sampleWeight?: Vector): unknown;
  transform(X: Matrix): Matrix;
  fitTransform?: (X: Matrix, y?: Vector, sampleWeight?: Vector) => Matrix;
  getParams?: (deep?: boolean) => Record<string, unknown>;
  setParams?: (params: Record<string, unknown>) => unknown;
}

export type FeatureUnionTransformer = TransformerLike | "drop" | "passthrough";
export type FeatureUnionSpec = [name: string, transformer: FeatureUnionTransformer];

export interface FeatureUnionOptions {
  transformerWeights?: Record<string, number>;
}

function fitTransform(
  transformer: TransformerLike,
  X: Matrix,
  y?: Vector,
  sampleWeight?: Vector,
): Matrix {
  if (typeof transformer.fitTransform === "function") {
    return transformer.fitTransform(X, y, sampleWeight);
  }
  transformer.fit(X, y, sampleWeight);
  return transformer.transform(X);
}

function applyWeight(block: Matrix, weight: number): Matrix {
  if (!Number.isFinite(weight)) {
    throw new Error(`FeatureUnion transformer weight must be finite. Got ${weight}.`);
  }
  if (weight === 1) {
    return block;
  }
  return block.map((row) => row.map((value) => value * weight));
}

function passthroughBlock(X: Matrix): Matrix {
  return X.map((row) => row.slice());
}

interface RuntimeSpec {
  name: string;
  transformer: FeatureUnionTransformer;
}

export class FeatureUnion {
  transformerList_: ReadonlyArray<readonly [string, FeatureUnionTransformer]> = [];
  namedTransformers_: Record<string, FeatureUnionTransformer> = {};

  private specs: FeatureUnionSpec[];
  private runtimeSpecs: RuntimeSpec[] = [];
  private nFeaturesIn: number | null = null;
  private isFitted = false;
  private transformerWeights: Record<string, number>;

  constructor(specs: FeatureUnionSpec[], options: FeatureUnionOptions = {}) {
    if (!Array.isArray(specs) || specs.length === 0) {
      throw new Error("FeatureUnion requires at least one transformer.");
    }
    this.specs = specs.map(([name, transformer]) => [name, transformer]);
    this.transformerWeights = { ...(options.transformerWeights ?? {}) };
    this.validateSpecs();
  }

  fit(X: Matrix, y?: Vector, sampleWeight?: Vector): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    this.nFeaturesIn = X[0].length;

    this.runtimeSpecs = [];
    for (let i = 0; i < this.specs.length; i += 1) {
      const [name, transformer] = this.specs[i];
      if (transformer !== "drop" && transformer !== "passthrough") {
        transformer.fit(X, y, sampleWeight);
      }
      this.runtimeSpecs.push({ name, transformer });
    }

    this.refreshViews();
    this.isFitted = true;
    return this;
  }

  transform(X: Matrix): Matrix {
    if (!this.isFitted || this.nFeaturesIn === null) {
      throw new Error("FeatureUnion has not been fitted.");
    }
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn}, got ${X[0].length}.`);
    }

    const blocks = this.runtimeSpecs.map(({ name, transformer }) => {
      if (transformer === "drop") {
        return Array.from({ length: X.length }, () => []);
      }
      if (transformer === "passthrough") {
        return applyWeight(passthroughBlock(X), this.transformerWeights[name] ?? 1);
      }
      return applyWeight(transformer.transform(X), this.transformerWeights[name] ?? 1);
    });

    const out = new Array<number[]>(X.length);
    for (let rowIndex = 0; rowIndex < X.length; rowIndex += 1) {
      const row: number[] = [];
      for (let b = 0; b < blocks.length; b += 1) {
        row.push(...blocks[b][rowIndex]);
      }
      out[rowIndex] = row;
    }
    return out;
  }

  fitTransform(X: Matrix, y?: Vector, sampleWeight?: Vector): Matrix {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    this.nFeaturesIn = X[0].length;

    this.runtimeSpecs = [];
    const blocks: Matrix[] = [];
    for (let i = 0; i < this.specs.length; i += 1) {
      const [name, transformer] = this.specs[i];
      let block: Matrix;
      if (transformer === "drop") {
        block = Array.from({ length: X.length }, () => []);
      } else if (transformer === "passthrough") {
        block = passthroughBlock(X);
      } else {
        block = fitTransform(transformer, X, y, sampleWeight);
      }
      blocks.push(applyWeight(block, this.transformerWeights[name] ?? 1));
      this.runtimeSpecs.push({ name, transformer });
    }

    this.refreshViews();
    this.isFitted = true;

    const out = new Array<number[]>(X.length);
    for (let rowIndex = 0; rowIndex < X.length; rowIndex += 1) {
      const row: number[] = [];
      for (let b = 0; b < blocks.length; b += 1) {
        row.push(...blocks[b][rowIndex]);
      }
      out[rowIndex] = row;
    }
    return out;
  }

  getParams(deep = true): Record<string, unknown> {
    const params: Record<string, unknown> = {
      transformerWeights: { ...this.transformerWeights },
    };
    for (let i = 0; i < this.specs.length; i += 1) {
      const [name, transformer] = this.specs[i];
      params[name] = transformer;
      params[`${name}__weight`] = this.transformerWeights[name] ?? 1;
      if (deep && transformer !== "drop" && transformer !== "passthrough" && transformer.getParams) {
        const nested = transformer.getParams(true);
        for (const [key, value] of Object.entries(nested)) {
          params[`${name}__${key}`] = value;
        }
      }
    }
    return params;
  }

  setParams(params: Record<string, unknown>): this {
    const nestedByName = new Map<string, Record<string, unknown>>();
    for (const [key, value] of Object.entries(params)) {
      if (key === "transformerWeights") {
        if (typeof value !== "object" || value === null) {
          throw new Error("transformerWeights must be an object map.");
        }
        this.transformerWeights = { ...(value as Record<string, number>) };
        this.isFitted = false;
        continue;
      }

      if (key.includes("__")) {
        const split = key.indexOf("__");
        const name = key.slice(0, split);
        const nestedKey = key.slice(split + 2);
        const bucket = nestedByName.get(name);
        if (bucket) {
          bucket[nestedKey] = value;
        } else {
          nestedByName.set(name, { [nestedKey]: value });
        }
        continue;
      }

      const spec = this.specs.find(([name]) => name === key);
      if (!spec) {
        throw new Error(`Unknown FeatureUnion parameter '${key}'.`);
      }
      if (value !== "drop" && value !== "passthrough" && typeof value !== "object") {
        throw new Error(
          `FeatureUnion step replacement for '${key}' must be 'drop', 'passthrough', or transformer object.`,
        );
      }
      spec[1] = value as FeatureUnionTransformer;
      this.isFitted = false;
    }

    for (const [name, nested] of nestedByName.entries()) {
      const spec = this.specs.find(([stepName]) => stepName === name);
      if (!spec) {
        throw new Error(`Unknown FeatureUnion step '${name}'.`);
      }

      if (nested.hasOwnProperty("weight")) {
        const weight = nested.weight;
        if (!Number.isFinite(weight as number)) {
          throw new Error(`FeatureUnion weight for '${name}' must be finite.`);
        }
        this.transformerWeights[name] = weight as number;
      }

      const transformer = spec[1];
      if (transformer === "drop" || transformer === "passthrough") {
        const keys = Object.keys(nested).filter((k) => k !== "weight");
        if (keys.length > 0) {
          throw new Error(`Cannot set nested params on FeatureUnion step '${name}' (${transformer}).`);
        }
      } else {
        const copy = { ...nested };
        delete copy.weight;
        if (Object.keys(copy).length > 0) {
          if (typeof transformer.setParams === "function") {
            transformer.setParams(copy);
          } else {
            for (const [k, v] of Object.entries(copy)) {
              (transformer as unknown as Record<string, unknown>)[k] = v;
            }
          }
        }
      }
      this.isFitted = false;
    }

    this.validateSpecs();
    this.refreshViews();
    return this;
  }

  private validateSpecs(): void {
    const seen = new Set<string>();
    for (let i = 0; i < this.specs.length; i += 1) {
      const [name] = this.specs[i];
      if (typeof name !== "string" || name.trim().length === 0) {
        throw new Error("FeatureUnion transformer names must be non-empty strings.");
      }
      if (seen.has(name)) {
        throw new Error(`FeatureUnion transformer names must be unique. Duplicate '${name}'.`);
      }
      seen.add(name);
    }
  }

  private refreshViews(): void {
    this.transformerList_ = this.runtimeSpecs.map(
      ({ name, transformer }) => [name, transformer] as const,
    );
    this.namedTransformers_ = Object.fromEntries(
      this.runtimeSpecs.map(({ name, transformer }) => [name, transformer]),
    );
  }
}
