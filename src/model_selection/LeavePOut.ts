import type { FoldIndices } from "./KFold";

export interface LeavePOutOptions {
  p?: number;
}

function buildCombinations(values: number[], choose: number): number[][] {
  const out: number[][] = [];
  const current: number[] = [];

  function visit(start: number, remaining: number): void {
    if (remaining === 0) {
      out.push(current.slice());
      return;
    }

    const maxStart = values.length - remaining;
    for (let i = start; i <= maxStart; i += 1) {
      current.push(values[i]!);
      visit(i + 1, remaining - 1);
      current.pop();
    }
  }

  visit(0, choose);
  return out;
}

export class LeavePOut {
  private readonly p: number;

  constructor(options: LeavePOutOptions = {}) {
    this.p = options.p ?? 2;
    if (!Number.isInteger(this.p) || this.p < 1) {
      throw new Error(`p must be an integer >= 1. Got ${this.p}.`);
    }
  }

  split<TX>(X: TX[], y?: unknown[]): FoldIndices[] {
    if (!Array.isArray(X) || X.length === 0) {
      throw new Error("X must be a non-empty array.");
    }
    if (y && y.length !== X.length) {
      throw new Error(`X and y must have the same length. Got ${X.length} and ${y.length}.`);
    }
    if (this.p >= X.length) {
      throw new Error(`p (${this.p}) must be smaller than sample count (${X.length}).`);
    }

    const sampleIndices = Array.from({ length: X.length }, (_, index) => index);
    const testCombinations = buildCombinations(sampleIndices, this.p);
    return testCombinations.map((testIndices) => {
      const testMask = new Uint8Array(X.length);
      for (let i = 0; i < testIndices.length; i += 1) {
        testMask[testIndices[i]!] = 1;
      }

      const trainIndices: number[] = [];
      for (let i = 0; i < X.length; i += 1) {
        if (testMask[i] === 0) {
          trainIndices.push(i);
        }
      }

      return { trainIndices, testIndices };
    });
  }

  getNSplits<TX>(X: TX[], y?: unknown[], _groups?: unknown[]): number {
    if (!Array.isArray(X) || X.length === 0) {
      throw new Error("X must be a non-empty array.");
    }
    if (y && y.length !== X.length) {
      throw new Error(`X and y must have the same length. Got ${X.length} and ${y.length}.`);
    }
    if (this.p >= X.length) {
      throw new Error(`p (${this.p}) must be smaller than sample count (${X.length}).`);
    }

    let numerator = 1;
    let denominator = 1;
    for (let i = 0; i < this.p; i += 1) {
      numerator *= X.length - i;
      denominator *= i + 1;
    }
    return numerator / denominator;
  }
}
