import type { Matrix, Vector } from "../types";
import { assertConsistentRowSize, assertFiniteMatrix, assertNonEmptyMatrix } from "../utils/validation";

interface TransformerLike {
  fit(X: Matrix, y?: Vector): unknown;
  transform(X: Matrix): Matrix;
  fitTransform?: (X: Matrix, y?: Vector) => Matrix;
}

export type FeatureUnionSpec = [name: string, transformer: TransformerLike];

function fitTransform(transformer: TransformerLike, X: Matrix, y?: Vector): Matrix {
  if (typeof transformer.fitTransform === "function") {
    return transformer.fitTransform(X, y);
  }
  transformer.fit(X, y);
  return transformer.transform(X);
}

export class FeatureUnion {
  transformerList_: ReadonlyArray<readonly [string, TransformerLike]> = [];

  private readonly specs: FeatureUnionSpec[];
  private runtimeSpecs: FeatureUnionSpec[] = [];
  private nFeaturesIn: number | null = null;
  private isFitted = false;

  constructor(specs: FeatureUnionSpec[]) {
    if (!Array.isArray(specs) || specs.length === 0) {
      throw new Error("FeatureUnion requires at least one transformer.");
    }
    this.specs = specs;
  }

  fit(X: Matrix, y?: Vector): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    this.nFeaturesIn = X[0].length;

    const seen = new Set<string>();
    this.runtimeSpecs = [];
    for (let i = 0; i < this.specs.length; i += 1) {
      const [name, transformer] = this.specs[i];
      if (typeof name !== "string" || name.trim().length === 0) {
        throw new Error("FeatureUnion transformer names must be non-empty strings.");
      }
      if (seen.has(name)) {
        throw new Error(`FeatureUnion transformer names must be unique. Duplicate '${name}'.`);
      }
      seen.add(name);
      transformer.fit(X, y);
      this.runtimeSpecs.push([name, transformer]);
    }

    this.transformerList_ = this.runtimeSpecs.map(
      ([name, transformer]) => [name, transformer] as const,
    );
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

    const blocks = this.runtimeSpecs.map(([, transformer]) => transformer.transform(X));
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

  fitTransform(X: Matrix, y?: Vector): Matrix {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    this.nFeaturesIn = X[0].length;

    const seen = new Set<string>();
    this.runtimeSpecs = [];
    const blocks: Matrix[] = [];
    for (let i = 0; i < this.specs.length; i += 1) {
      const [name, transformer] = this.specs[i];
      if (typeof name !== "string" || name.trim().length === 0) {
        throw new Error("FeatureUnion transformer names must be non-empty strings.");
      }
      if (seen.has(name)) {
        throw new Error(`FeatureUnion transformer names must be unique. Duplicate '${name}'.`);
      }
      seen.add(name);
      blocks.push(fitTransform(transformer, X, y));
      this.runtimeSpecs.push([name, transformer]);
    }

    this.transformerList_ = this.runtimeSpecs.map(
      ([name, transformer]) => [name, transformer] as const,
    );
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
}
