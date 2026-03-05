import type { FoldIndices } from "./KFold";

export interface ShuffleSplitOptions {
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
    const tmp = values[i]!;
    values[i] = values[j]!;
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

export class ShuffleSplit {
  private readonly nSplits: number;
  private readonly testSize?: number;
  private readonly trainSize?: number;
  private readonly randomState: number;

  constructor(options: ShuffleSplitOptions = {}) {
    this.nSplits = options.nSplits ?? 10;
    this.testSize = options.testSize;
    this.trainSize = options.trainSize;
    this.randomState = options.randomState ?? 42;

    if (!Number.isInteger(this.nSplits) || this.nSplits < 1) {
      throw new Error(`nSplits must be an integer >= 1. Got ${this.nSplits}.`);
    }
  }

  split<TX>(X: TX[], y?: unknown[]): FoldIndices[] {
    if (!Array.isArray(X) || X.length === 0) {
      throw new Error("X must be a non-empty array.");
    }
    if (y && y.length !== X.length) {
      throw new Error(`X and y must have the same length. Got ${X.length} and ${y.length}.`);
    }

    const { trainCount, testCount } = resolveSplitCounts(X.length, this.testSize, this.trainSize);
    if (trainCount < 1 || testCount < 1) {
      throw new Error("Both train and test sets must have at least one sample.");
    }

    const splits: FoldIndices[] = [];
    for (let splitIndex = 0; splitIndex < this.nSplits; splitIndex += 1) {
      const shuffledIndices = Array.from({ length: X.length }, (_, index) => index);
      const random = mulberry32(this.randomState + splitIndex * 104_729);
      shuffleInPlace(shuffledIndices, random);

      const testIndices = shuffledIndices.slice(0, testCount).sort((left, right) => left - right);
      const trainIndices = shuffledIndices
        .slice(testCount, testCount + trainCount)
        .sort((left, right) => left - right);

      splits.push({ trainIndices, testIndices });
    }
    return splits;
  }

  getNSplits(): number {
    return this.nSplits;
  }
}
