import type { FoldIndices } from "./KFold";

export class LeaveOneOut {
  split<TX>(X: TX[], y?: unknown[]): FoldIndices[] {
    if (!Array.isArray(X) || X.length === 0) {
      throw new Error("X must be a non-empty array.");
    }
    if (y && y.length !== X.length) {
      throw new Error(`X and y must have the same length. Got ${X.length} and ${y.length}.`);
    }

    const folds: FoldIndices[] = [];
    for (let testIndex = 0; testIndex < X.length; testIndex += 1) {
      const trainIndices: number[] = [];
      for (let i = 0; i < X.length; i += 1) {
        if (i !== testIndex) {
          trainIndices.push(i);
        }
      }
      folds.push({ trainIndices, testIndices: [testIndex] });
    }
    return folds;
  }

  getNSplits<TX>(X: TX[], y?: unknown[], _groups?: unknown[]): number {
    if (!Array.isArray(X) || X.length === 0) {
      throw new Error("X must be a non-empty array.");
    }
    if (y && y.length !== X.length) {
      throw new Error(`X and y must have the same length. Got ${X.length} and ${y.length}.`);
    }
    return X.length;
  }
}
