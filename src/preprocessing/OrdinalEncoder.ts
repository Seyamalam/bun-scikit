import type { Matrix } from "../types";
import { assertConsistentRowSize, assertFiniteMatrix, assertNonEmptyMatrix } from "../utils/validation";

export interface OrdinalEncoderOptions {
  handleUnknown?: "error" | "use_encoded_value";
  unknownValue?: number;
}

export class OrdinalEncoder {
  categories_: number[][] | null = null;
  nFeaturesIn_: number | null = null;

  private readonly handleUnknown: "error" | "use_encoded_value";
  private readonly unknownValue: number;

  constructor(options: OrdinalEncoderOptions = {}) {
    this.handleUnknown = options.handleUnknown ?? "error";
    this.unknownValue = options.unknownValue ?? -1;
    if (!Number.isFinite(this.unknownValue)) {
      throw new Error(`unknownValue must be finite. Got ${this.unknownValue}.`);
    }
  }

  fit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    const nFeatures = X[0].length;
    const categories = new Array<number[]>(nFeatures);
    for (let feature = 0; feature < nFeatures; feature += 1) {
      const unique = new Set<number>();
      for (let i = 0; i < X.length; i += 1) {
        unique.add(X[i][feature]);
      }
      categories[feature] = Array.from(unique).sort((a, b) => a - b);
    }

    this.categories_ = categories;
    this.nFeaturesIn_ = nFeatures;
    return this;
  }

  transform(X: Matrix): Matrix {
    if (this.categories_ === null || this.nFeaturesIn_ === null) {
      throw new Error("OrdinalEncoder has not been fitted.");
    }
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }

    const out = new Array<number[]>(X.length);
    for (let i = 0; i < X.length; i += 1) {
      const row = new Array<number>(this.nFeaturesIn_);
      for (let feature = 0; feature < this.nFeaturesIn_; feature += 1) {
        const categoryIndex = this.categories_[feature].indexOf(X[i][feature]);
        if (categoryIndex === -1) {
          if (this.handleUnknown === "error") {
            throw new Error(`Unknown category ${X[i][feature]} in feature ${feature}.`);
          }
          row[feature] = this.unknownValue;
        } else {
          row[feature] = categoryIndex;
        }
      }
      out[i] = row;
    }
    return out;
  }

  inverseTransform(X: Matrix): Matrix {
    if (this.categories_ === null || this.nFeaturesIn_ === null) {
      throw new Error("OrdinalEncoder has not been fitted.");
    }
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }

    const out = new Array<number[]>(X.length);
    for (let i = 0; i < X.length; i += 1) {
      const row = new Array<number>(this.nFeaturesIn_);
      for (let feature = 0; feature < this.nFeaturesIn_; feature += 1) {
        const encoded = X[i][feature];
        if (encoded === this.unknownValue && this.handleUnknown === "use_encoded_value") {
          row[feature] = this.unknownValue;
          continue;
        }
        if (!Number.isInteger(encoded) || encoded < 0 || encoded >= this.categories_[feature].length) {
          throw new Error(
            `Encoded value ${encoded} is out of range for feature ${feature}.`,
          );
        }
        row[feature] = this.categories_[feature][encoded];
      }
      out[i] = row;
    }
    return out;
  }

  fitTransform(X: Matrix): Matrix {
    return this.fit(X).transform(X);
  }
}
