import type { Matrix, Vector } from "../types";
import { assertFiniteVector } from "../utils/validation";

export interface LabelBinarizerOptions {
  negLabel?: number;
  posLabel?: number;
  sparseOutput?: boolean;
}

export class LabelBinarizer {
  classes_: Vector | null = null;
  yType_: "binary" | "multiclass" | null = null;

  private negLabel: number;
  private posLabel: number;
  private sparseOutput: boolean;
  private fitted = false;

  constructor(options: LabelBinarizerOptions = {}) {
    this.negLabel = options.negLabel ?? 0;
    this.posLabel = options.posLabel ?? 1;
    this.sparseOutput = options.sparseOutput ?? false;
    if (!Number.isFinite(this.negLabel) || !Number.isFinite(this.posLabel)) {
      throw new Error("negLabel and posLabel must be finite numbers.");
    }
    if (this.negLabel === this.posLabel) {
      throw new Error("negLabel and posLabel must differ.");
    }
  }

  fit(y: Vector): this {
    if (!Array.isArray(y) || y.length === 0) {
      throw new Error("y must be a non-empty array.");
    }
    assertFiniteVector(y);
    const classes = Array.from(new Set(y)).sort((a, b) => a - b);
    this.classes_ = classes;
    this.yType_ = classes.length <= 2 ? "binary" : "multiclass";
    this.fitted = true;
    return this;
  }

  transform(y: Vector): Matrix {
    this.assertFitted();
    assertFiniteVector(y);

    const classToIndex = new Map<number, number>();
    for (let i = 0; i < this.classes_!.length; i += 1) {
      classToIndex.set(this.classes_![i], i);
    }

    if (this.yType_ === "binary") {
      const positiveClass = this.classes_![this.classes_!.length - 1];
      const out: Matrix = new Array(y.length);
      for (let i = 0; i < y.length; i += 1) {
        if (!classToIndex.has(y[i])) {
          throw new Error(`Unknown label ${y[i]} at index ${i}.`);
        }
        out[i] = [y[i] === positiveClass ? this.posLabel : this.negLabel];
      }
      return out;
    }

    const out: Matrix = Array.from({ length: y.length }, () => new Array<number>(this.classes_!.length).fill(this.negLabel));
    for (let i = 0; i < y.length; i += 1) {
      const index = classToIndex.get(y[i]);
      if (index === undefined) {
        throw new Error(`Unknown label ${y[i]} at index ${i}.`);
      }
      out[i][index] = this.posLabel;
    }
    return out;
  }

  fitTransform(y: Vector): Matrix {
    return this.fit(y).transform(y);
  }

  inverseTransform(Y: Matrix): Vector {
    this.assertFitted();
    if (!Array.isArray(Y) || Y.length === 0) {
      throw new Error("Y must be a non-empty matrix.");
    }

    if (this.yType_ === "binary") {
      const positiveClass = this.classes_![this.classes_!.length - 1];
      const negativeClass = this.classes_![0];
      const out = new Array<number>(Y.length).fill(negativeClass);
      for (let i = 0; i < Y.length; i += 1) {
        if (Y[i].length !== 1) {
          throw new Error("Binary inverseTransform expects a single output column.");
        }
        out[i] = Y[i][0] === this.posLabel ? positiveClass : negativeClass;
      }
      return out;
    }

    const out = new Array<number>(Y.length).fill(0);
    for (let i = 0; i < Y.length; i += 1) {
      if (Y[i].length !== this.classes_!.length) {
        throw new Error(`Feature size mismatch. Expected ${this.classes_!.length}, got ${Y[i].length}.`);
      }
      let bestIndex = 0;
      let bestValue = Y[i][0];
      for (let j = 1; j < Y[i].length; j += 1) {
        if (Y[i][j] > bestValue) {
          bestValue = Y[i][j];
          bestIndex = j;
        }
      }
      out[i] = this.classes_![bestIndex];
    }
    return out;
  }

  get isSparseOutput(): boolean {
    return this.sparseOutput;
  }

  private assertFitted(): void {
    if (!this.fitted || this.classes_ === null || this.yType_ === null) {
      throw new Error("LabelBinarizer has not been fitted.");
    }
  }
}

