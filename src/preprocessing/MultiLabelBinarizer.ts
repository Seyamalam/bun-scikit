import type { Matrix } from "../types";

export type MultiLabel = string | number;

export interface MultiLabelBinarizerOptions {
  classes?: MultiLabel[] | null;
  sparseOutput?: boolean;
}

function labelSort(a: MultiLabel, b: MultiLabel): number {
  if (typeof a === "number" && typeof b === "number") {
    return a - b;
  }
  return String(a).localeCompare(String(b));
}

function labelKey(value: MultiLabel): string {
  return typeof value === "number" ? `n:${value}` : `s:${value}`;
}

export class MultiLabelBinarizer {
  classes_: MultiLabel[] | null = null;

  private classesOption: MultiLabel[] | null;
  private sparseOutput: boolean;
  private classToIndex: Map<string, number> = new Map();
  private fitted = false;

  constructor(options: MultiLabelBinarizerOptions = {}) {
    this.classesOption = options.classes ?? null;
    this.sparseOutput = options.sparseOutput ?? false;
  }

  fit(y: MultiLabel[][]): this {
    if (!Array.isArray(y) || y.length === 0) {
      throw new Error("y must be a non-empty array of iterables.");
    }

    let classes: MultiLabel[];
    if (this.classesOption) {
      classes = this.classesOption.slice();
    } else {
      const classSet = new Map<string, MultiLabel>();
      for (let i = 0; i < y.length; i += 1) {
        for (let j = 0; j < y[i].length; j += 1) {
          const value = y[i][j];
          classSet.set(labelKey(value), value);
        }
      }
      classes = Array.from(classSet.values()).sort(labelSort);
    }

    this.classToIndex = new Map();
    for (let i = 0; i < classes.length; i += 1) {
      this.classToIndex.set(labelKey(classes[i]), i);
    }
    this.classes_ = classes;
    this.fitted = true;
    return this;
  }

  transform(y: MultiLabel[][]): Matrix {
    this.assertFitted();
    if (!Array.isArray(y) || y.length === 0) {
      throw new Error("y must be a non-empty array of iterables.");
    }

    const out: Matrix = Array.from({ length: y.length }, () => new Array<number>(this.classes_!.length).fill(0));
    for (let i = 0; i < y.length; i += 1) {
      for (let j = 0; j < y[i].length; j += 1) {
        const index = this.classToIndex.get(labelKey(y[i][j]));
        if (index === undefined) {
          throw new Error(`Unknown class '${String(y[i][j])}' encountered during transform.`);
        }
        out[i][index] = 1;
      }
    }
    return out;
  }

  fitTransform(y: MultiLabel[][]): Matrix {
    return this.fit(y).transform(y);
  }

  inverseTransform(Y: Matrix): MultiLabel[][] {
    this.assertFitted();
    if (!Array.isArray(Y) || Y.length === 0) {
      throw new Error("Y must be a non-empty matrix.");
    }
    const out: MultiLabel[][] = new Array(Y.length);
    for (let i = 0; i < Y.length; i += 1) {
      if (Y[i].length !== this.classes_!.length) {
        throw new Error(`Feature size mismatch. Expected ${this.classes_!.length}, got ${Y[i].length}.`);
      }
      const labels: MultiLabel[] = [];
      for (let j = 0; j < Y[i].length; j += 1) {
        if (Y[i][j] > 0) {
          labels.push(this.classes_![j]);
        }
      }
      out[i] = labels;
    }
    return out;
  }

  get isSparseOutput(): boolean {
    return this.sparseOutput;
  }

  private assertFitted(): void {
    if (!this.fitted || this.classes_ === null) {
      throw new Error("MultiLabelBinarizer has not been fitted.");
    }
  }
}

