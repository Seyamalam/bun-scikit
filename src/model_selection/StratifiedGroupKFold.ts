import type { FoldIndices } from "./KFold";

export interface StratifiedGroupKFoldOptions {
  nSplits?: number;
  shuffle?: boolean;
  randomState?: number;
}

function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffleInPlace<T>(values: T[], random: () => number): void {
  for (let i = values.length - 1; i > 0; i -= 1) {
    const j = Math.floor(random() * (i + 1));
    const tmp = values[i];
    values[i] = values[j];
    values[j] = tmp;
  }
}

interface GroupInfo {
  group: number;
  indices: number[];
  classCounts: number[];
  size: number;
}

function computeImbalance(
  foldClassCounts: number[][],
  classTotals: number[],
  nSplits: number,
): number {
  let total = 0;
  for (let classIndex = 0; classIndex < classTotals.length; classIndex += 1) {
    const denom = Math.max(1, classTotals[classIndex]);
    let mean = 0;
    for (let fold = 0; fold < nSplits; fold += 1) {
      mean += foldClassCounts[fold][classIndex] / denom;
    }
    mean /= nSplits;

    let variance = 0;
    for (let fold = 0; fold < nSplits; fold += 1) {
      const normalized = foldClassCounts[fold][classIndex] / denom;
      const diff = normalized - mean;
      variance += diff * diff;
    }
    total += variance / nSplits;
  }
  return total;
}

export class StratifiedGroupKFold {
  private readonly nSplits: number;
  private readonly shuffle: boolean;
  private readonly randomState: number;

  constructor(options: StratifiedGroupKFoldOptions = {}) {
    this.nSplits = options.nSplits ?? 5;
    this.shuffle = options.shuffle ?? false;
    this.randomState = options.randomState ?? 42;
    if (!Number.isInteger(this.nSplits) || this.nSplits < 2) {
      throw new Error(`nSplits must be an integer >= 2. Got ${this.nSplits}.`);
    }
  }

  split<TX>(X: TX[], y: number[], groups?: number[]): FoldIndices[] {
    if (!Array.isArray(X) || X.length === 0) {
      throw new Error("X must be a non-empty array.");
    }
    if (!Array.isArray(y) || y.length !== X.length) {
      throw new Error(`X and y must have the same length. Got ${X.length} and ${y.length}.`);
    }
    if (!Array.isArray(groups) || groups.length !== X.length) {
      throw new Error(
        `StratifiedGroupKFold requires groups with same length as X. Got ${groups?.length ?? 0} and ${X.length}.`,
      );
    }

    const uniqueClasses = Array.from(new Set(y)).sort((a, b) => a - b);
    if (uniqueClasses.length < 2) {
      throw new Error("StratifiedGroupKFold requires at least two distinct classes.");
    }

    const classToIndex = new Map<number, number>();
    for (let i = 0; i < uniqueClasses.length; i += 1) {
      classToIndex.set(uniqueClasses[i], i);
    }

    const groupMap = new Map<number, GroupInfo>();
    for (let i = 0; i < X.length; i += 1) {
      const group = groups[i];
      const classIndex = classToIndex.get(y[i]);
      if (classIndex === undefined) {
        throw new Error(`Internal StratifiedGroupKFold error for class ${y[i]}.`);
      }
      let info = groupMap.get(group);
      if (!info) {
        info = {
          group,
          indices: [],
          classCounts: new Array<number>(uniqueClasses.length).fill(0),
          size: 0,
        };
        groupMap.set(group, info);
      }
      info.indices.push(i);
      info.classCounts[classIndex] += 1;
      info.size += 1;
    }

    const groupInfos = Array.from(groupMap.values());
    if (groupInfos.length < this.nSplits) {
      throw new Error(
        `nSplits (${this.nSplits}) cannot exceed unique group count (${groupInfos.length}).`,
      );
    }

    if (this.shuffle) {
      const random = mulberry32(this.randomState);
      shuffleInPlace(groupInfos, random);
    }
    groupInfos.sort((a, b) => {
      const bySize = b.size - a.size;
      if (bySize !== 0) {
        return bySize;
      }
      let maxDelta = 0;
      for (let i = 0; i < a.classCounts.length; i += 1) {
        maxDelta = Math.max(maxDelta, Math.abs(a.classCounts[i] - b.classCounts[i]));
      }
      if (maxDelta !== 0) {
        return maxDelta;
      }
      return a.group - b.group;
    });

    const classTotals = new Array<number>(uniqueClasses.length).fill(0);
    for (let i = 0; i < y.length; i += 1) {
      const classIndex = classToIndex.get(y[i])!;
      classTotals[classIndex] += 1;
    }

    const foldGroups: number[][] = Array.from({ length: this.nSplits }, () => []);
    const foldSizes = new Array<number>(this.nSplits).fill(0);
    const foldClassCounts: number[][] = Array.from({ length: this.nSplits }, () =>
      new Array<number>(uniqueClasses.length).fill(0),
    );

    for (let i = 0; i < groupInfos.length; i += 1) {
      const info = groupInfos[i];
      let bestFold = 0;
      let bestObjective = Number.POSITIVE_INFINITY;
      for (let fold = 0; fold < this.nSplits; fold += 1) {
        for (let classIndex = 0; classIndex < uniqueClasses.length; classIndex += 1) {
          foldClassCounts[fold][classIndex] += info.classCounts[classIndex];
        }
        foldSizes[fold] += info.size;

        const imbalance = computeImbalance(foldClassCounts, classTotals, this.nSplits);
        const sizePenalty = foldSizes[fold] / Math.max(1, X.length);
        const objective = imbalance + sizePenalty * 1e-6;

        foldSizes[fold] -= info.size;
        for (let classIndex = 0; classIndex < uniqueClasses.length; classIndex += 1) {
          foldClassCounts[fold][classIndex] -= info.classCounts[classIndex];
        }

        if (objective < bestObjective) {
          bestObjective = objective;
          bestFold = fold;
        }
      }

      foldGroups[bestFold].push(info.group);
      foldSizes[bestFold] += info.size;
      for (let classIndex = 0; classIndex < uniqueClasses.length; classIndex += 1) {
        foldClassCounts[bestFold][classIndex] += info.classCounts[classIndex];
      }
    }

    const groupsByFold = new Map<number, number>();
    for (let fold = 0; fold < foldGroups.length; fold += 1) {
      const assigned = foldGroups[fold];
      for (let i = 0; i < assigned.length; i += 1) {
        groupsByFold.set(assigned[i], fold);
      }
    }

    const folds: FoldIndices[] = Array.from({ length: this.nSplits }, () => ({
      trainIndices: [],
      testIndices: [],
    }));

    for (let i = 0; i < groups.length; i += 1) {
      const fold = groupsByFold.get(groups[i]);
      if (fold === undefined) {
        throw new Error(
          `Internal StratifiedGroupKFold error: missing fold assignment for group ${groups[i]}.`,
        );
      }
      folds[fold].testIndices.push(i);
    }

    for (let fold = 0; fold < folds.length; fold += 1) {
      const testMask = new Uint8Array(X.length);
      const testIndices = folds[fold].testIndices;
      for (let i = 0; i < testIndices.length; i += 1) {
        testMask[testIndices[i]] = 1;
      }
      const trainIndices = folds[fold].trainIndices;
      for (let i = 0; i < X.length; i += 1) {
        if (testMask[i] === 0) {
          trainIndices.push(i);
        }
      }
      if (folds[fold].testIndices.length === 0 || folds[fold].trainIndices.length === 0) {
        throw new Error(
          "StratifiedGroupKFold produced an empty fold. Consider reducing nSplits.",
        );
      }
    }

    return folds;
  }
}
