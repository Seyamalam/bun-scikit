export interface FoldIndices {
  trainIndices: number[];
  testIndices: number[];
}

export interface KFoldOptions {
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

function shuffleInPlace(indices: number[], seed: number): void {
  const random = mulberry32(seed);
  for (let i = indices.length - 1; i > 0; i -= 1) {
    const j = Math.floor(random() * (i + 1));
    const tmp = indices[i];
    indices[i] = indices[j];
    indices[j] = tmp;
  }
}

export class KFold {
  private readonly nSplits: number;
  private readonly shuffle: boolean;
  private readonly randomState: number;

  constructor(options: KFoldOptions = {}) {
    this.nSplits = options.nSplits ?? 5;
    this.shuffle = options.shuffle ?? false;
    this.randomState = options.randomState ?? 42;

    if (!Number.isInteger(this.nSplits) || this.nSplits < 2) {
      throw new Error(`nSplits must be an integer >= 2. Got ${this.nSplits}.`);
    }
  }

  split<TX>(X: TX[], y?: unknown[]): FoldIndices[] {
    if (!Array.isArray(X) || X.length === 0) {
      throw new Error("X must be a non-empty array.");
    }
    if (y && y.length !== X.length) {
      throw new Error(`X and y must have the same length. Got ${X.length} and ${y.length}.`);
    }
    if (this.nSplits > X.length) {
      throw new Error(`nSplits (${this.nSplits}) cannot exceed sample count (${X.length}).`);
    }

    const indices = Array.from({ length: X.length }, (_, idx) => idx);
    if (this.shuffle) {
      shuffleInPlace(indices, this.randomState);
    }

    const foldSizes = new Array<number>(this.nSplits).fill(Math.floor(X.length / this.nSplits));
    const remainder = X.length % this.nSplits;
    for (let i = 0; i < remainder; i += 1) {
      foldSizes[i] += 1;
    }

    const folds: FoldIndices[] = [];
    let start = 0;
    for (let foldIdx = 0; foldIdx < this.nSplits; foldIdx += 1) {
      const size = foldSizes[foldIdx];
      const end = start + size;
      const testIndices = indices.slice(start, end);
      const trainIndices = indices.slice(0, start).concat(indices.slice(end));
      folds.push({ trainIndices, testIndices });
      start = end;
    }

    return folds;
  }
}
