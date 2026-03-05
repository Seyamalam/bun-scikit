import type { FoldIndices } from "./KFold";

export interface LeavePGroupsOutOptions {
  p?: number;
}

function collectUniqueGroups(groups: number[]): number[] {
  const uniqueGroups: number[] = [];
  const seen = new Set<number>();
  for (let i = 0; i < groups.length; i += 1) {
    const group = groups[i]!;
    if (!seen.has(group)) {
      seen.add(group);
      uniqueGroups.push(group);
    }
  }
  return uniqueGroups;
}

function buildGroupCombinations(values: number[], choose: number): number[][] {
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

export class LeavePGroupsOut {
  private readonly p: number;

  constructor(options: LeavePGroupsOutOptions = {}) {
    this.p = options.p ?? 2;
    if (!Number.isInteger(this.p) || this.p < 1) {
      throw new Error(`p must be an integer >= 1. Got ${this.p}.`);
    }
  }

  split<TX>(X: TX[], _y?: unknown[], groups?: number[]): FoldIndices[] {
    if (!Array.isArray(X) || X.length === 0) {
      throw new Error("X must be a non-empty array.");
    }
    if (!Array.isArray(groups) || groups.length !== X.length) {
      throw new Error(
        `LeavePGroupsOut requires groups with same length as X. Got ${groups?.length ?? 0} and ${X.length}.`,
      );
    }

    const uniqueGroups = collectUniqueGroups(groups);
    if (uniqueGroups.length <= this.p) {
      throw new Error(`p (${this.p}) must be smaller than unique group count (${uniqueGroups.length}).`);
    }

    const heldOutGroupSets = buildGroupCombinations(uniqueGroups, this.p);
    return heldOutGroupSets.map((heldOutGroups) => {
      const heldOutSet = new Set(heldOutGroups);
      const trainIndices: number[] = [];
      const testIndices: number[] = [];

      for (let i = 0; i < groups.length; i += 1) {
        if (heldOutSet.has(groups[i]!)) {
          testIndices.push(i);
        } else {
          trainIndices.push(i);
        }
      }

      return { trainIndices, testIndices };
    });
  }

  getNSplits<TX>(X: TX[], _y?: unknown[], groups?: number[]): number {
    if (!Array.isArray(X) || X.length === 0) {
      throw new Error("X must be a non-empty array.");
    }
    if (!Array.isArray(groups) || groups.length !== X.length) {
      throw new Error(
        `LeavePGroupsOut requires groups with same length as X. Got ${groups?.length ?? 0} and ${X.length}.`,
      );
    }

    const uniqueGroups = collectUniqueGroups(groups);
    if (uniqueGroups.length <= this.p) {
      throw new Error(`p (${this.p}) must be smaller than unique group count (${uniqueGroups.length}).`);
    }

    let numerator = 1;
    let denominator = 1;
    for (let i = 0; i < this.p; i += 1) {
      numerator *= uniqueGroups.length - i;
      denominator *= i + 1;
    }
    return numerator / denominator;
  }
}
