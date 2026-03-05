import type { FoldIndices } from "./KFold";

export interface PredefinedSplitOptions {
  testFold: number[];
}

function resolveTestFoldInput(input: number[] | PredefinedSplitOptions): number[] {
  if (Array.isArray(input)) {
    return input.slice();
  }
  return input.testFold.slice();
}

export class PredefinedSplit {
  private readonly testFold: number[];

  constructor(testFold: number[] | PredefinedSplitOptions) {
    this.testFold = resolveTestFoldInput(testFold);
    if (this.testFold.length === 0) {
      throw new Error("testFold must be a non-empty array.");
    }
    for (let i = 0; i < this.testFold.length; i += 1) {
      const fold = this.testFold[i]!;
      if (!Number.isInteger(fold) || fold < -1) {
        throw new Error(`testFold values must be integers >= -1. Got ${fold} at index ${i}.`);
      }
    }
  }

  split<TX>(X: TX[], y?: unknown[]): FoldIndices[] {
    if (!Array.isArray(X) || X.length !== this.testFold.length) {
      throw new Error(
        `X must have the same length as testFold. Got ${X.length} and ${this.testFold.length}.`,
      );
    }
    if (y && y.length !== X.length) {
      throw new Error(`X and y must have the same length. Got ${X.length} and ${y.length}.`);
    }

    const uniqueFolds = Array.from(
      new Set(this.testFold.filter((fold) => fold >= 0)),
    ).sort((left, right) => left - right);

    return uniqueFolds.map((foldId) => {
      const trainIndices: number[] = [];
      const testIndices: number[] = [];

      for (let i = 0; i < this.testFold.length; i += 1) {
        if (this.testFold[i] === foldId) {
          testIndices.push(i);
        } else {
          trainIndices.push(i);
        }
      }

      return { trainIndices, testIndices };
    });
  }

  getNSplits(): number {
    return new Set(this.testFold.filter((fold) => fold >= 0)).size;
  }
}
