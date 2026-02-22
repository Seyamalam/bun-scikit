import type { Matrix, Vector } from "../types";
import { assertConsistentRowSize, assertFiniteMatrix, assertNonEmptyMatrix } from "../utils/validation";

interface TransformerLike {
  fit(X: Matrix, y?: Vector): unknown;
  transform(X: Matrix): Matrix;
  fitTransform?: (X: Matrix, y?: Vector) => Matrix;
}

export type ColumnSelector = number[] | { start: number; end: number };

export type ColumnTransformerSpec = [name: string, transformer: TransformerLike, columns: ColumnSelector];

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

function fitTransform(transformer: TransformerLike, X: Matrix, y?: Vector): Matrix {
  if (typeof transformer.fitTransform === "function") {
    return transformer.fitTransform(X, y);
  }
  transformer.fit(X, y);
  return transformer.transform(X);
}

interface RuntimeSpec {
  name: string;
  transformer: TransformerLike;
  columns: number[];
}

export class ColumnTransformer {
  transformers_: ReadonlyArray<readonly [string, TransformerLike, number[]]> = [];

  private readonly specs: ColumnTransformerSpec[];
  private readonly remainder: "drop" | "passthrough";
  private runtimeSpecs: RuntimeSpec[] = [];
  private passthroughColumns: number[] = [];
  private nFeaturesIn: number | null = null;
  private isFitted = false;

  constructor(specs: ColumnTransformerSpec[], options: ColumnTransformerOptions = {}) {
    if (!Array.isArray(specs) || specs.length === 0) {
      throw new Error("ColumnTransformer requires at least one transformer spec.");
    }
    this.specs = specs;
    this.remainder = options.remainder ?? "drop";
  }

  fit(X: Matrix, y?: Vector): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    const featureCount = X[0].length;
    this.nFeaturesIn = featureCount;
    const used = new Uint8Array(featureCount);
    this.runtimeSpecs = [];

    for (let i = 0; i < this.specs.length; i += 1) {
      const [name, transformer, selector] = this.specs[i];
      if (typeof name !== "string" || name.trim().length === 0) {
        throw new Error("ColumnTransformer spec names must be non-empty strings.");
      }
      const columns = normalizeColumns(selector, featureCount);
      for (let j = 0; j < columns.length; j += 1) {
        used[columns[j]] = 1;
      }
      const subX = selectColumns(X, columns);
      transformer.fit(subX, y);
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

    this.transformers_ = this.runtimeSpecs.map(
      (spec) => [spec.name, spec.transformer, spec.columns] as const,
    );
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

    const transformedBlocks = this.runtimeSpecs.map((spec) =>
      spec.transformer.transform(selectColumns(X, spec.columns)),
    );
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

  fitTransform(X: Matrix, y?: Vector): Matrix {
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
      const transformed = fitTransform(transformer, selectColumns(X, columns), y);
      transformedBlocks.push(transformed);
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

    this.transformers_ = this.runtimeSpecs.map(
      (spec) => [spec.name, spec.transformer, spec.columns] as const,
    );
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
}
