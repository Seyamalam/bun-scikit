import type { Matrix } from "../types";
import { assertConsistentRowSize, assertFiniteMatrix, assertNonEmptyMatrix } from "../utils/validation";

export interface OneHotEncoderOptions {
  handleUnknown?: "error" | "ignore";
}

export class OneHotEncoder {
  categories_: number[][] | null = null;
  nFeaturesIn_: number | null = null;
  nOutputFeatures_: number | null = null;
  featureOffsets_: number[] | null = null;

  private readonly handleUnknown: "error" | "ignore";

  constructor(options: OneHotEncoderOptions = {}) {
    this.handleUnknown = options.handleUnknown ?? "error";
  }

  fit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    const nFeatures = X[0].length;
    const categories = new Array<number[]>(nFeatures);
    const offsets = new Array<number>(nFeatures);
    let offset = 0;

    for (let featureIndex = 0; featureIndex < nFeatures; featureIndex += 1) {
      const unique = new Set<number>();
      for (let i = 0; i < X.length; i += 1) {
        unique.add(X[i][featureIndex]);
      }
      const values = Array.from(unique).sort((a, b) => a - b);
      categories[featureIndex] = values;
      offsets[featureIndex] = offset;
      offset += values.length;
    }

    this.categories_ = categories;
    this.nFeaturesIn_ = nFeatures;
    this.nOutputFeatures_ = offset;
    this.featureOffsets_ = offsets;
    return this;
  }

  transform(X: Matrix): Matrix {
    if (
      this.categories_ === null ||
      this.nFeaturesIn_ === null ||
      this.nOutputFeatures_ === null ||
      this.featureOffsets_ === null
    ) {
      throw new Error("OneHotEncoder has not been fitted.");
    }

    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }

    const encoded = new Array<number[]>(X.length);
    for (let rowIndex = 0; rowIndex < X.length; rowIndex += 1) {
      const row = new Array<number>(this.nOutputFeatures_).fill(0);
      for (let featureIndex = 0; featureIndex < this.nFeaturesIn_; featureIndex += 1) {
        const value = X[rowIndex][featureIndex];
        const categories = this.categories_[featureIndex];
        const categoryIndex = categories.indexOf(value);
        if (categoryIndex === -1) {
          if (this.handleUnknown === "error") {
            throw new Error(
              `Unknown category ${value} in feature ${featureIndex}. Set handleUnknown='ignore' to skip.`,
            );
          }
          continue;
        }
        row[this.featureOffsets_[featureIndex] + categoryIndex] = 1;
      }
      encoded[rowIndex] = row;
    }
    return encoded;
  }

  fitTransform(X: Matrix): Matrix {
    return this.fit(X).transform(X);
  }
}
