import type { FoldIndices } from "./KFold";

export interface StratifiedKFoldOptions {
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

function shuffleInPlace(indices: number[], random: () => number): void {
  for (let i = indices.length - 1; i > 0; i -= 1) {
    const j = Math.floor(random() * (i + 1));
    const tmp = indices[i];
    indices[i] = indices[j];
    indices[j] = tmp;
  }
}

export class StratifiedKFold {
  private readonly nSplits: number;
  private readonly shuffle: boolean;
  private readonly randomState: number;

  constructor(options: StratifiedKFoldOptions = {}) {
    this.nSplits = options.nSplits ?? 5;
    this.shuffle = options.shuffle ?? false;
    this.randomState = options.randomState ?? 42;

    if (!Number.isInteger(this.nSplits) || this.nSplits < 2) {
      throw new Error(`nSplits must be an integer >= 2. Got ${this.nSplits}.`);
    }
  }

  split<TX>(X: TX[], y: number[]): FoldIndices[] {
    if (!Array.isArray(X) || X.length === 0) {
      throw new Error("X must be a non-empty array.");
    }
    if (!Array.isArray(y) || y.length !== X.length) {
      throw new Error(`X and y must have the same length. Got ${X.length} and ${y.length}.`);
    }
    if (this.nSplits > X.length) {
      throw new Error(`nSplits (${this.nSplits}) cannot exceed sample count (${X.length}).`);
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
      throw new Error("StratifiedKFold requires at least two distinct classes.");
    }

    let minClassCount = Number.POSITIVE_INFINITY;
    for (const indices of byClass.values()) {
      if (indices.length < minClassCount) {
        minClassCount = indices.length;
      }
    }
    if (minClassCount < this.nSplits) {
      throw new Error(
        `nSplits (${this.nSplits}) cannot exceed the smallest class count (${minClassCount}).`,
      );
    }

    const foldTestIndices = Array.from({ length: this.nSplits }, () => new Array<number>());
    const random = mulberry32(this.randomState);

    for (const classIndices of byClass.values()) {
      const working = classIndices.slice();
      if (this.shuffle) {
        shuffleInPlace(working, random);
      }
      for (let i = 0; i < working.length; i += 1) {
        foldTestIndices[i % this.nSplits].push(working[i]);
      }
    }

    const folds: FoldIndices[] = [];
    for (let foldIdx = 0; foldIdx < this.nSplits; foldIdx += 1) {
      const testIndices = foldTestIndices[foldIdx].slice().sort((a, b) => a - b);
      const testMask = new Uint8Array(X.length);
      for (let i = 0; i < testIndices.length; i += 1) {
        testMask[testIndices[i]] = 1;
      }
      const trainIndices = new Array<number>();
      for (let i = 0; i < X.length; i += 1) {
        if (testMask[i] === 0) {
          trainIndices.push(i);
        }
      }
      folds.push({ trainIndices, testIndices });
    }

    return folds;
  }
}
