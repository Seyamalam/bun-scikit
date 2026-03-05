import type { FoldIndices } from "./KFold";

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

export class LeaveOneGroupOut {
  split<TX>(X: TX[], _y?: unknown[], groups?: number[]): FoldIndices[] {
    if (!Array.isArray(X) || X.length === 0) {
      throw new Error("X must be a non-empty array.");
    }
    if (!Array.isArray(groups) || groups.length !== X.length) {
      throw new Error(
        `LeaveOneGroupOut requires groups with same length as X. Got ${groups?.length ?? 0} and ${X.length}.`,
      );
    }

    const uniqueGroups = collectUniqueGroups(groups);
    if (uniqueGroups.length < 2) {
      throw new Error("LeaveOneGroupOut requires at least two unique groups.");
    }

    return uniqueGroups.map((heldOutGroup) => {
      const trainIndices: number[] = [];
      const testIndices: number[] = [];

      for (let i = 0; i < groups.length; i += 1) {
        if (groups[i] === heldOutGroup) {
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
        `LeaveOneGroupOut requires groups with same length as X. Got ${groups?.length ?? 0} and ${X.length}.`,
      );
    }
    return collectUniqueGroups(groups).length;
  }
}
