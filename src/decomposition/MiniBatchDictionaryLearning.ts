import type { Matrix } from "../types";
import {
  DictionaryLearning,
  type DictionaryLearningOptions,
} from "./DictionaryLearning";

export interface MiniBatchDictionaryLearningOptions extends DictionaryLearningOptions {
  batchSize?: number;
}

export class MiniBatchDictionaryLearning extends DictionaryLearning {
  private batchSize: number;

  constructor(options: MiniBatchDictionaryLearningOptions = {}) {
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
