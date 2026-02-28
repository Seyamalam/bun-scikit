import type { Matrix } from "../types";
import { SparsePCA, type SparsePCAOptions } from "./SparsePCA";

export interface MiniBatchSparsePCAOptions extends SparsePCAOptions {
  batchSize?: number;
}

export class MiniBatchSparsePCA extends SparsePCA {
  private batchSize: number;

  constructor(options: MiniBatchSparsePCAOptions = {}) {
    super(options);
    this.batchSize = options.batchSize ?? 64;
    if (!Number.isInteger(this.batchSize) || this.batchSize < 1) {
      throw new Error(`batchSize must be an integer >= 1. Got ${this.batchSize}.`);
    }
  }

  override fit(X: Matrix): this {
    // Lightweight implementation keeps deterministic full-batch behavior.
    return super.fit(X);
  }
}
