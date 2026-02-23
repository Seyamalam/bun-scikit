import type { Vector } from "../types";
import { assertFiniteVector } from "../utils/validation";

export class LabelEncoder {
  classes_: number[] | null = null;
  private classToIndex: Map<number, number> | null = null;

  fit(y: Vector): this {
    if (!Array.isArray(y) || y.length === 0) {
      throw new Error("y must be a non-empty array.");
    }
    assertFiniteVector(y);

    const classes = Array.from(new Set(y)).sort((a, b) => a - b);
    const classToIndex = new Map<number, number>();
    for (let i = 0; i < classes.length; i += 1) {
      classToIndex.set(classes[i], i);
    }

    this.classes_ = classes;
    this.classToIndex = classToIndex;
    return this;
  }

  transform(y: Vector): Vector {
    if (!this.classToIndex) {
      throw new Error("LabelEncoder has not been fitted.");
    }
    assertFiniteVector(y);

    const encoded = new Array<number>(y.length);
    for (let i = 0; i < y.length; i += 1) {
      const idx = this.classToIndex.get(y[i]);
      if (idx === undefined) {
        throw new Error(`Unknown label ${y[i]} at index ${i}.`);
      }
      encoded[i] = idx;
    }
    return encoded;
  }

  fitTransform(y: Vector): Vector {
    return this.fit(y).transform(y);
  }

  inverseTransform(y: Vector): Vector {
    if (!this.classes_) {
      throw new Error("LabelEncoder has not been fitted.");
    }
    assertFiniteVector(y);

    const decoded = new Array<number>(y.length);
    for (let i = 0; i < y.length; i += 1) {
      const encoded = y[i];
      if (!Number.isInteger(encoded) || encoded < 0 || encoded >= this.classes_.length) {
        throw new Error(`Encoded label out of range at index ${i}: ${encoded}.`);
      }
      decoded[i] = this.classes_[encoded];
    }
    return decoded;
  }
}
