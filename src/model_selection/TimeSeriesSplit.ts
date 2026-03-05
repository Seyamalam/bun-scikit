import type { FoldIndices } from "./KFold";

export interface TimeSeriesSplitOptions {
  nSplits?: number;
  maxTrainSize?: number;
  testSize?: number;
  gap?: number;
}

export class TimeSeriesSplit {
  private readonly nSplits: number;
  private readonly maxTrainSize?: number;
  private readonly testSize?: number;
  private readonly gap: number;

  constructor(options: TimeSeriesSplitOptions = {}) {
    this.nSplits = options.nSplits ?? 5;
    this.maxTrainSize = options.maxTrainSize;
    this.testSize = options.testSize;
    this.gap = options.gap ?? 0;

    if (!Number.isInteger(this.nSplits) || this.nSplits < 2) {
      throw new Error(`nSplits must be an integer >= 2. Got ${this.nSplits}.`);
    }
    if (
      this.maxTrainSize !== undefined &&
      (!Number.isInteger(this.maxTrainSize) || this.maxTrainSize < 1)
    ) {
      throw new Error(`maxTrainSize must be an integer >= 1. Got ${this.maxTrainSize}.`);
    }
    if (this.testSize !== undefined && (!Number.isInteger(this.testSize) || this.testSize < 1)) {
      throw new Error(`testSize must be an integer >= 1. Got ${this.testSize}.`);
    }
    if (!Number.isInteger(this.gap) || this.gap < 0) {
      throw new Error(`gap must be an integer >= 0. Got ${this.gap}.`);
    }
  }

  split<TX>(X: TX[], y?: unknown[]): FoldIndices[] {
    if (!Array.isArray(X) || X.length === 0) {
      throw new Error("X must be a non-empty array.");
    }
    if (y && y.length !== X.length) {
      throw new Error(`X and y must have the same length. Got ${X.length} and ${y.length}.`);
    }

    const sampleCount = X.length;
    const resolvedTestSize = this.testSize ?? Math.floor((sampleCount - this.gap) / (this.nSplits + 1));
    if (resolvedTestSize < 1) {
      throw new Error("TimeSeriesSplit could not resolve a positive testSize.");
    }

    const initialTrainSize = sampleCount - this.nSplits * resolvedTestSize;
    if (initialTrainSize < 1) {
      throw new Error(
        `Too many splits (${this.nSplits}) for sample count ${sampleCount} with testSize ${resolvedTestSize}.`,
      );
    }

    const folds: FoldIndices[] = [];
    for (let splitIndex = 0; splitIndex < this.nSplits; splitIndex += 1) {
      const testStart = initialTrainSize + splitIndex * resolvedTestSize;
      const testEnd = testStart + resolvedTestSize;
      const trainEnd = testStart - this.gap;

      if (trainEnd < 1 || testEnd > sampleCount) {
        throw new Error(
          `Invalid time-series split boundaries for split ${splitIndex}: trainEnd=${trainEnd}, testEnd=${testEnd}, samples=${sampleCount}.`,
        );
      }

      const trainStart =
        this.maxTrainSize === undefined ? 0 : Math.max(0, trainEnd - this.maxTrainSize);

      const trainIndices: number[] = [];
      for (let i = trainStart; i < trainEnd; i += 1) {
        trainIndices.push(i);
      }

      const testIndices: number[] = [];
      for (let i = testStart; i < testEnd; i += 1) {
        testIndices.push(i);
      }

      folds.push({ trainIndices, testIndices });
    }

    return folds;
  }

  getNSplits(): number {
    return this.nSplits;
  }
}
