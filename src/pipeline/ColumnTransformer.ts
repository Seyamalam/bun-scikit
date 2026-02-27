import type { Matrix, Vector } from "../types";
import { assertConsistentRowSize, assertFiniteMatrix, assertNonEmptyMatrix } from "../utils/validation";

interface TransformerLike {
  fit(X: Matrix, y?: Vector, sampleWeight?: Vector): unknown;
  transform(X: Matrix): Matrix;
  fitTransform?: (X: Matrix, y?: Vector, sampleWeight?: Vector) => Matrix;
  getParams?: (deep?: boolean) => Record<string, unknown>;
  setParams?: (params: Record<string, unknown>) => unknown;
}

export type ColumnSelector = number[] | { start: number; end: number };
export type ColumnTransformerStep = TransformerLike | "drop" | "passthrough";
export type ColumnTransformerSpec = [
  name: string,
  transformer: ColumnTransformerStep,
  columns: ColumnSelector,
];

export interface ColumnTransformerOptions {
  remainder?: "drop" | "passthrough";
}

function normalizeColumns(selector: ColumnSelector, featureCount: number): number[] {
  if (Array.isArray(selector)) {
    if (selector.length === 0) {
      throw new Error("Column selector arrays must not be empty.");
    }
    const normalized = selector.map((index) => {
      if (!Number.isInteger(index) || index < 0 || index >= featureCount) {
        throw new Error(`Invalid column index ${index} for featureCount=${featureCount}.`);
      }
      return index;
    });
    return Array.from(new Set(normalized)).sort((a, b) => a - b);
  }

  const { start, end } = selector;
  if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end <= start || end > featureCount) {
    throw new Error(
      `Range selector must satisfy 0 <= start < end <= featureCount. Got start=${start}, end=${end}, featureCount=${featureCount}.`,
    );
  }
  const columns = new Array<number>(end - start);
  for (let i = start; i < end; i += 1) {
    columns[i - start] = i;
  }
  return columns;
}

function selectColumns(X: Matrix, columns: number[]): Matrix {
  return X.map((row) => columns.map((idx) => row[idx]));
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

function cloneSpec(spec: ColumnTransformerSpec): ColumnTransformerSpec {
  return [spec[0], spec[1], Array.isArray(spec[2]) ? spec[2].slice() : { ...spec[2] }];
}

interface RuntimeSpec {
  name: string;
  transformer: ColumnTransformerStep;
  columns: number[];
}

export class ColumnTransformer {
  transformers_: ReadonlyArray<readonly [string, ColumnTransformerStep, number[]]> = [];
  namedTransformers_: Record<string, ColumnTransformerStep> = {};

  private specs: ColumnTransformerSpec[];
  private remainder: "drop" | "passthrough";
  private runtimeSpecs: RuntimeSpec[] = [];
  private passthroughColumns: number[] = [];
  private nFeaturesIn: number | null = null;
  private isFitted = false;

  constructor(specs: ColumnTransformerSpec[], options: ColumnTransformerOptions = {}) {
    if (!Array.isArray(specs) || specs.length === 0) {
      throw new Error("ColumnTransformer requires at least one transformer spec.");
    }
    this.specs = specs.map(cloneSpec);
    this.remainder = options.remainder ?? "drop";
    this.validateSpecNames();
  }

  fit(X: Matrix, y?: Vector, sampleWeight?: Vector): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    const featureCount = X[0].length;
    this.nFeaturesIn = featureCount;
    const used = new Uint8Array(featureCount);
    this.runtimeSpecs = [];

    for (let i = 0; i < this.specs.length; i += 1) {
      const [name, transformer, selector] = this.specs[i];
      const columns = normalizeColumns(selector, featureCount);
      for (let j = 0; j < columns.length; j += 1) {
        used[columns[j]] = 1;
      }
      if (transformer !== "drop" && transformer !== "passthrough") {
        const subX = selectColumns(X, columns);
        transformer.fit(subX, y, sampleWeight);
      }
      this.runtimeSpecs.push({ name, transformer, columns });
    }

    this.passthroughColumns = [];
    if (this.remainder === "passthrough") {
      for (let i = 0; i < featureCount; i += 1) {
        if (used[i] === 0) {
          this.passthroughColumns.push(i);
        }
      }
    }

    this.refreshViews();
    this.isFitted = true;
    return this;
  }

  transform(X: Matrix): Matrix {
    if (!this.isFitted || this.nFeaturesIn === null) {
      throw new Error("ColumnTransformer has not been fitted.");
    }
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn}, got ${X[0].length}.`);
    }

    const transformedBlocks = this.runtimeSpecs.map((spec) => {
      const subX = selectColumns(X, spec.columns);
      if (spec.transformer === "drop") {
        return Array.from({ length: X.length }, () => []);
      }
      if (spec.transformer === "passthrough") {
        return subX;
      }
      return spec.transformer.transform(subX);
    });
    const passthroughBlock =
      this.passthroughColumns.length > 0 ? selectColumns(X, this.passthroughColumns) : null;

    const out = new Array<number[]>(X.length);
    for (let rowIndex = 0; rowIndex < X.length; rowIndex += 1) {
      const row: number[] = [];
      for (let b = 0; b < transformedBlocks.length; b += 1) {
        row.push(...transformedBlocks[b][rowIndex]);
      }
      if (passthroughBlock) {
        row.push(...passthroughBlock[rowIndex]);
      }
      out[rowIndex] = row;
    }
    return out;
  }

  fitTransform(X: Matrix, y?: Vector, sampleWeight?: Vector): Matrix {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    const featureCount = X[0].length;
    this.nFeaturesIn = featureCount;
    const used = new Uint8Array(featureCount);
    this.runtimeSpecs = [];

    const transformedBlocks: Matrix[] = [];
    for (let i = 0; i < this.specs.length; i += 1) {
      const [name, transformer, selector] = this.specs[i];
      const columns = normalizeColumns(selector, featureCount);
      for (let j = 0; j < columns.length; j += 1) {
        used[columns[j]] = 1;
      }
      const subX = selectColumns(X, columns);

      if (transformer === "drop") {
        transformedBlocks.push(Array.from({ length: X.length }, () => []));
      } else if (transformer === "passthrough") {
        transformedBlocks.push(subX);
      } else {
        transformedBlocks.push(fitTransform(transformer, subX, y, sampleWeight));
      }
      this.runtimeSpecs.push({ name, transformer, columns });
    }

    this.passthroughColumns = [];
    if (this.remainder === "passthrough") {
      for (let i = 0; i < featureCount; i += 1) {
        if (used[i] === 0) {
          this.passthroughColumns.push(i);
        }
      }
    }
    const passthroughBlock =
      this.passthroughColumns.length > 0 ? selectColumns(X, this.passthroughColumns) : null;

    this.refreshViews();
    this.isFitted = true;

    const out = new Array<number[]>(X.length);
    for (let rowIndex = 0; rowIndex < X.length; rowIndex += 1) {
      const row: number[] = [];
      for (let b = 0; b < transformedBlocks.length; b += 1) {
        row.push(...transformedBlocks[b][rowIndex]);
      }
      if (passthroughBlock) {
        row.push(...passthroughBlock[rowIndex]);
      }
      out[rowIndex] = row;
    }
    return out;
  }

  getParams(deep = true): Record<string, unknown> {
    const params: Record<string, unknown> = { remainder: this.remainder };
    for (let i = 0; i < this.specs.length; i += 1) {
      const [name, transformer, columns] = this.specs[i];
      params[name] = transformer;
      params[`${name}__columns`] = columns;
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
      if (key === "remainder") {
        if (value !== "drop" && value !== "passthrough") {
          throw new Error("remainder must be 'drop' or 'passthrough'.");
        }
        this.remainder = value;
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
        throw new Error(`Unknown ColumnTransformer parameter '${key}'.`);
      }
      if (value !== "drop" && value !== "passthrough" && typeof value !== "object") {
        throw new Error(
          `ColumnTransformer step replacement for '${key}' must be 'drop', 'passthrough', or a transformer object.`,
        );
      }
      spec[1] = value as ColumnTransformerStep;
      this.isFitted = false;
    }

    for (const [name, nested] of nestedByName.entries()) {
      const spec = this.specs.find(([stepName]) => stepName === name);
      if (!spec) {
        throw new Error(`Unknown ColumnTransformer step '${name}'.`);
      }
      if (nested.hasOwnProperty("columns")) {
        spec[2] = nested.columns as ColumnSelector;
      }

      const transformer = spec[1];
      if (transformer === "drop" || transformer === "passthrough") {
        const keys = Object.keys(nested).filter((k) => k !== "columns");
        if (keys.length > 0) {
          throw new Error(`Cannot set nested params on ColumnTransformer step '${name}' (${transformer}).`);
        }
      } else {
        const copy = { ...nested };
        delete copy.columns;
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

    this.validateSpecNames();
    this.refreshViews();
    return this;
  }

  private validateSpecNames(): void {
    const seen = new Set<string>();
    for (let i = 0; i < this.specs.length; i += 1) {
      const [name] = this.specs[i];
      if (typeof name !== "string" || name.trim().length === 0) {
        throw new Error("ColumnTransformer spec names must be non-empty strings.");
      }
      if (seen.has(name)) {
        throw new Error(`ColumnTransformer spec names must be unique. Duplicate '${name}'.`);
      }
      seen.add(name);
    }
  }

  private refreshViews(): void {
    this.transformers_ = this.runtimeSpecs.map(
      (spec) => [spec.name, spec.transformer, spec.columns] as const,
    );
    this.namedTransformers_ = Object.fromEntries(this.runtimeSpecs.map((spec) => [spec.name, spec.transformer]));
  }
}
