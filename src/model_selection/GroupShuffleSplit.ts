import type { FoldIndices } from "./KFold";

export interface GroupShuffleSplitOptions {
  nSplits?: number;
  testSize?: number;
  trainSize?: number;
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

function shuffleInPlace(values: number[], random: () => number): void {
  for (let i = values.length - 1; i > 0; i -= 1) {
    const j = Math.floor(random() * (i + 1));
    const tmp = values[i];
    values[i] = values[j];
    values[j] = tmp;
  }
}

function resolveSplitCounts(
  groupCount: number,
  testSize: number | undefined,
  trainSize: number | undefined,
): { trainGroupCount: number; testGroupCount: number } {
  function resolveSize(value: number | undefined, label: string): number | null {
    if (value === undefined) {
      return null;
    }
    if (value > 0 && value < 1) {
      return Math.max(1, Math.floor(groupCount * value));
    }
    if (Number.isInteger(value) && value >= 1 && value < groupCount) {
      return value;
    }
    throw new Error(
      `${label} must be a float in (0, 1) or int in [1, n_groups-1]. Got ${value}.`,
    );
  }

  const resolvedTest = resolveSize(testSize, "testSize");
  const resolvedTrain = resolveSize(trainSize, "trainSize");

  if (resolvedTest === null && resolvedTrain === null) {
    const defaultTest = Math.max(1, Math.floor(groupCount * 0.2));
    return {
      testGroupCount: defaultTest,
      trainGroupCount: groupCount - defaultTest,
    };
  }

  if (resolvedTest !== null && resolvedTrain !== null) {
    if (resolvedTest + resolvedTrain > groupCount) {
      throw new Error(
        `trainSize + testSize must be <= unique group count (${groupCount}). Got ${resolvedTrain + resolvedTest}.`,
      );
    }
    return { trainGroupCount: resolvedTrain, testGroupCount: resolvedTest };
  }

  if (resolvedTest !== null) {
    return { testGroupCount: resolvedTest, trainGroupCount: groupCount - resolvedTest };
  }

  const trainGroupCount = resolvedTrain!;
  return { trainGroupCount, testGroupCount: groupCount - trainGroupCount };
}

export class GroupShuffleSplit {
  private readonly nSplits: number;
  private readonly testSize?: number;
  private readonly trainSize?: number;
  private readonly randomState: number;

  constructor(options: GroupShuffleSplitOptions = {}) {
    this.nSplits = options.nSplits ?? 5;
    this.testSize = options.testSize;
    this.trainSize = options.trainSize;
    this.randomState = options.randomState ?? 42;

    if (!Number.isInteger(this.nSplits) || this.nSplits < 1) {
      throw new Error(`nSplits must be an integer >= 1. Got ${this.nSplits}.`);
    }
  }

  split<TX>(X: TX[], _y?: unknown[], groups?: number[]): FoldIndices[] {
    if (!Array.isArray(X) || X.length === 0) {
      throw new Error("X must be a non-empty array.");
    }
    if (!Array.isArray(groups) || groups.length !== X.length) {
      throw new Error(
        `GroupShuffleSplit requires groups with same length as X. Got ${groups?.length ?? 0} and ${X.length}.`,
      );
    }

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
    const uniqueGroups = Array.from(byGroup.keys());
    if (uniqueGroups.length < 2) {
      throw new Error("GroupShuffleSplit requires at least two unique groups.");
    }

    const { trainGroupCount, testGroupCount } = resolveSplitCounts(
      uniqueGroups.length,
      this.testSize,
      this.trainSize,
    );
    if (trainGroupCount < 1 || testGroupCount < 1) {
      throw new Error("Both train and test group sets must have at least one group.");
    }

    const splits: FoldIndices[] = [];
    for (let splitIndex = 0; splitIndex < this.nSplits; splitIndex += 1) {
      const shuffledGroups = uniqueGroups.slice();
      const random = mulberry32(this.randomState + splitIndex * 104_729);
      shuffleInPlace(shuffledGroups, random);

      const testGroups = new Set<number>(shuffledGroups.slice(0, testGroupCount));
      const trainIndices: number[] = [];
      const testIndices: number[] = [];

      for (let i = 0; i < groups.length; i += 1) {
        if (testGroups.has(groups[i])) {
          testIndices.push(i);
        } else {
          trainIndices.push(i);
        }
      }

      if (trainIndices.length === 0 || testIndices.length === 0) {
        throw new Error(
          "GroupShuffleSplit produced an empty train or test split. Adjust trainSize/testSize.",
        );
      }
      splits.push({ trainIndices, testIndices });
    }
    return splits;
  }
}
