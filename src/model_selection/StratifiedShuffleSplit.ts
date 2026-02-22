import type { FoldIndices } from "./KFold";

export interface StratifiedShuffleSplitOptions {
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
  sampleCount: number,
  testSize: number | undefined,
  trainSize: number | undefined,
): { trainCount: number; testCount: number } {
  function resolveSize(value: number | undefined, label: string): number | null {
    if (value === undefined) {
      return null;
    }
    if (value > 0 && value < 1) {
      return Math.max(1, Math.floor(sampleCount * value));
    }
    if (Number.isInteger(value) && value >= 1 && value < sampleCount) {
      return value;
    }
    throw new Error(
      `${label} must be a float in (0, 1) or int in [1, n-1]. Got ${value}.`,
    );
  }

  const resolvedTest = resolveSize(testSize, "testSize");
  const resolvedTrain = resolveSize(trainSize, "trainSize");

  if (resolvedTest === null && resolvedTrain === null) {
    const defaultTest = Math.max(1, Math.floor(sampleCount * 0.1));
    return {
      testCount: defaultTest,
      trainCount: sampleCount - defaultTest,
    };
  }

  if (resolvedTest !== null && resolvedTrain !== null) {
    if (resolvedTest + resolvedTrain > sampleCount) {
      throw new Error(
        `trainSize + testSize must be <= sample count (${sampleCount}). Got ${resolvedTrain + resolvedTest}.`,
      );
    }
    return { trainCount: resolvedTrain, testCount: resolvedTest };
  }

  if (resolvedTest !== null) {
    return { testCount: resolvedTest, trainCount: sampleCount - resolvedTest };
  }

  const trainCount = resolvedTrain!;
  return { trainCount, testCount: sampleCount - trainCount };
}

export class StratifiedShuffleSplit {
  private readonly nSplits: number;
  private readonly testSize?: number;
  private readonly trainSize?: number;
  private readonly randomState: number;

  constructor(options: StratifiedShuffleSplitOptions = {}) {
    this.nSplits = options.nSplits ?? 10;
    this.testSize = options.testSize;
    this.trainSize = options.trainSize;
    this.randomState = options.randomState ?? 42;

    if (!Number.isInteger(this.nSplits) || this.nSplits < 1) {
      throw new Error(`nSplits must be an integer >= 1. Got ${this.nSplits}.`);
    }
  }

  split<TX>(X: TX[], y: number[]): FoldIndices[] {
    if (!Array.isArray(X) || X.length === 0) {
      throw new Error("X must be a non-empty array.");
    }
    if (!Array.isArray(y) || y.length !== X.length) {
      throw new Error(`X and y must have the same length. Got ${X.length} and ${y.length}.`);
    }

    const { trainCount, testCount } = resolveSplitCounts(X.length, this.testSize, this.trainSize);
    if (trainCount < 1 || testCount < 1) {
      throw new Error("Both train and test sets must have at least one sample.");
    }

    const byClass = new Map<number, number[]>();
    for (let i = 0; i < y.length; i += 1) {
      const label = y[i];
      const bucket = byClass.get(label);
      if (bucket) {
        bucket.push(i);
      } else {
        byClass.set(label, [i]);
      }
    }

    if (byClass.size < 2) {
      throw new Error("StratifiedShuffleSplit requires at least two classes.");
    }

    const classEntries = Array.from(byClass.entries());
    let minClassCount = Number.POSITIVE_INFINITY;
    for (const [, indices] of classEntries) {
      if (indices.length < minClassCount) {
        minClassCount = indices.length;
      }
    }
    if (minClassCount < 2) {
      throw new Error(
        "The least populated class has fewer than 2 members, which is not enough for stratified splitting.",
      );
    }

    const proportions = classEntries.map(([, indices]) => (indices.length * testCount) / X.length);
    const testPerClass = proportions.map((value) => Math.floor(value));

    // Respect the global test count while keeping per-class allocations feasible.
    let allocated = testPerClass.reduce((sum, count) => sum + count, 0);
    let remaining = testCount - allocated;
    if (remaining > 0) {
      const classOrder = proportions
        .map((target, idx) => ({ idx, frac: target - Math.floor(target) }))
        .sort((a, b) => b.frac - a.frac)
        .map((entry) => entry.idx);

      let cursor = 0;
      while (remaining > 0) {
        const classIdx = classOrder[cursor % classOrder.length];
        const classCount = classEntries[classIdx][1].length;
        if (testPerClass[classIdx] < classCount - 1) {
          testPerClass[classIdx] += 1;
          remaining -= 1;
        }
        cursor += 1;
      }
    }

    for (let i = 0; i < testPerClass.length; i += 1) {
      const classCount = classEntries[i][1].length;
      if (testPerClass[i] >= classCount) {
        testPerClass[i] = classCount - 1;
      }
    }

    allocated = testPerClass.reduce((sum, count) => sum + count, 0);
    if (allocated !== testCount) {
      throw new Error(
        `Could not allocate exactly ${testCount} stratified test samples. Allocated ${allocated}.`,
      );
    }

    const splits: FoldIndices[] = [];
    for (let splitIndex = 0; splitIndex < this.nSplits; splitIndex += 1) {
      const random = mulberry32(this.randomState + splitIndex * 104_729);
      const testIndices: number[] = [];

      for (let classIdx = 0; classIdx < classEntries.length; classIdx += 1) {
        const classIndices = classEntries[classIdx][1].slice();
        shuffleInPlace(classIndices, random);
        const classTestCount = testPerClass[classIdx];
        for (let i = 0; i < classTestCount; i += 1) {
          testIndices.push(classIndices[i]);
        }
      }

      testIndices.sort((a, b) => a - b);
      const testMask = new Uint8Array(X.length);
      for (let i = 0; i < testIndices.length; i += 1) {
        testMask[testIndices[i]] = 1;
      }

      const trainIndices: number[] = [];
      for (let i = 0; i < X.length; i += 1) {
        if (testMask[i] === 0) {
          trainIndices.push(i);
        }
      }

      if (trainIndices.length !== trainCount || testIndices.length !== testCount) {
        throw new Error(
          `Split sizes mismatch. Expected train/test ${trainCount}/${testCount}, got ${trainIndices.length}/${testIndices.length}.`,
        );
      }

      splits.push({ trainIndices, testIndices });
    }

    return splits;
  }
}
