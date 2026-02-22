import type { FoldIndices } from "./KFold";
import { StratifiedKFold } from "./StratifiedKFold";

export interface RepeatedStratifiedKFoldOptions {
  nSplits?: number;
  nRepeats?: number;
  randomState?: number;
}

export class RepeatedStratifiedKFold {
  private readonly nSplits: number;
  private readonly nRepeats: number;
  private readonly randomState: number;

  constructor(options: RepeatedStratifiedKFoldOptions = {}) {
    this.nSplits = options.nSplits ?? 5;
    this.nRepeats = options.nRepeats ?? 10;
    this.randomState = options.randomState ?? 42;

    if (!Number.isInteger(this.nSplits) || this.nSplits < 2) {
      throw new Error(`nSplits must be an integer >= 2. Got ${this.nSplits}.`);
    }
    if (!Number.isInteger(this.nRepeats) || this.nRepeats < 1) {
      throw new Error(`nRepeats must be an integer >= 1. Got ${this.nRepeats}.`);
    }
  }

  split<TX>(X: TX[], y: number[]): FoldIndices[] {
    if (!Array.isArray(X) || X.length === 0) {
      throw new Error("X must be a non-empty array.");
    }
    if (!Array.isArray(y) || y.length !== X.length) {
      throw new Error(`X and y must have the same length. Got ${X.length} and ${y.length}.`);
    }

    const allFolds: FoldIndices[] = [];
    for (let repeat = 0; repeat < this.nRepeats; repeat += 1) {
      const splitter = new StratifiedKFold({
        nSplits: this.nSplits,
        shuffle: true,
        randomState: this.randomState + repeat * 104_729,
      });
      const folds = splitter.split(X, y);
      for (let i = 0; i < folds.length; i += 1) {
        allFolds.push(folds[i]);
      }
    }
    return allFolds;
  }
}
