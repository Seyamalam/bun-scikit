import type { FoldIndices } from "./KFold";

export interface GroupKFoldOptions {
  nSplits?: number;
}

interface GroupBucket {
  group: number;
  indices: number[];
}

function buildGroupBuckets(groups: number[]): GroupBucket[] {
  const byGroup = new Map<number, number[]>();
  for (let i = 0; i < groups.length; i += 1) {
    const group = groups[i];
    const bucket = byGroup.get(group);
    if (bucket) {
      bucket.push(i);
    } else {
      byGroup.set(group, [i]);
    }
  }

  const out: GroupBucket[] = [];
  for (const [group, indices] of byGroup.entries()) {
    out.push({ group, indices });
  }
  out.sort((a, b) => {
    const bySize = b.indices.length - a.indices.length;
    if (bySize !== 0) {
      return bySize;
    }
    return a.group - b.group;
  });
  return out;
}

export class GroupKFold {
  private readonly nSplits: number;

  constructor(options: GroupKFoldOptions = {}) {
    this.nSplits = options.nSplits ?? 5;
    if (!Number.isInteger(this.nSplits) || this.nSplits < 2) {
      throw new Error(`nSplits must be an integer >= 2. Got ${this.nSplits}.`);
    }
  }

  split<TX>(X: TX[], _y?: unknown[], groups?: number[]): FoldIndices[] {
    if (!Array.isArray(X) || X.length === 0) {
      throw new Error("X must be a non-empty array.");
    }
    if (!Array.isArray(groups) || groups.length !== X.length) {
      throw new Error(
        `GroupKFold requires groups with same length as X. Got ${groups?.length ?? 0} and ${X.length}.`,
      );
    }
    if (this.nSplits > X.length) {
      throw new Error(`nSplits (${this.nSplits}) cannot exceed sample count (${X.length}).`);
    }

    const buckets = buildGroupBuckets(groups);
    if (buckets.length < this.nSplits) {
      throw new Error(
        `nSplits (${this.nSplits}) cannot exceed unique group count (${buckets.length}).`,
      );
    }

    const foldGroups: number[][] = Array.from({ length: this.nSplits }, () => []);
    const foldSizes = new Array<number>(this.nSplits).fill(0);
    for (let i = 0; i < buckets.length; i += 1) {
      let bestFold = 0;
      for (let fold = 1; fold < this.nSplits; fold += 1) {
        if (foldSizes[fold] < foldSizes[bestFold]) {
          bestFold = fold;
        }
      }
      foldGroups[bestFold].push(buckets[i].group);
      foldSizes[bestFold] += buckets[i].indices.length;
    }

    const groupsByFold = new Map<number, number>();
    for (let fold = 0; fold < foldGroups.length; fold += 1) {
      const current = foldGroups[fold];
      for (let i = 0; i < current.length; i += 1) {
        groupsByFold.set(current[i], fold);
      }
    }

    const folds: FoldIndices[] = Array.from({ length: this.nSplits }, () => ({
      trainIndices: [],
      testIndices: [],
    }));

    for (let i = 0; i < groups.length; i += 1) {
      const fold = groupsByFold.get(groups[i]);
      if (fold === undefined) {
        throw new Error(`Internal GroupKFold error: missing fold assignment for group ${groups[i]}.`);
      }
      folds[fold].testIndices.push(i);
    }

    for (let fold = 0; fold < folds.length; fold += 1) {
      const testMask = new Uint8Array(groups.length);
      const testIndices = folds[fold].testIndices;
      for (let i = 0; i < testIndices.length; i += 1) {
        testMask[testIndices[i]] = 1;
      }
      const trainIndices = folds[fold].trainIndices;
      for (let i = 0; i < groups.length; i += 1) {
        if (testMask[i] === 0) {
          trainIndices.push(i);
        }
      }
    }

    return folds;
  }
}
