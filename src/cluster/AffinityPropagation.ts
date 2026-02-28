import type { Matrix, Vector } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";

export interface AffinityPropagationOptions {
  damping?: number;
  maxIter?: number;
  convergenceIter?: number;
  preference?: number;
  randomState?: number;
}

function squaredDistance(a: Vector, b: Vector): number {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

function median(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }
  const sorted = values.slice().sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) {
    return 0.5 * (sorted[mid - 1] + sorted[mid]);
  }
  return sorted[mid];
}

function argmax(values: Vector): number {
  let bestIndex = 0;
  let bestValue = values[0];
  for (let i = 1; i < values.length; i += 1) {
    if (values[i] > bestValue) {
      bestValue = values[i];
      bestIndex = i;
    }
  }
  return bestIndex;
}

export class AffinityPropagation {
  clusterCentersIndices_: Vector | null = null;
  clusterCenters_: Matrix | null = null;
  labels_: Vector | null = null;
  nIter_: number | null = null;
  nFeaturesIn_: number | null = null;

  private damping: number;
  private maxIter: number;
  private convergenceIter: number;
  private preference?: number;
  private randomState?: number;
  private fitted = false;

  constructor(options: AffinityPropagationOptions = {}) {
    this.damping = options.damping ?? 0.5;
    this.maxIter = options.maxIter ?? 200;
    this.convergenceIter = options.convergenceIter ?? 15;
    this.preference = options.preference;
    this.randomState = options.randomState;

    if (!Number.isFinite(this.damping) || this.damping < 0.5 || this.damping < 0 || this.damping >= 1) {
      throw new Error(`damping must be in [0.5, 1). Got ${this.damping}.`);
    }
    if (!Number.isInteger(this.maxIter) || this.maxIter < 1) {
      throw new Error(`maxIter must be an integer >= 1. Got ${this.maxIter}.`);
    }
    if (!Number.isInteger(this.convergenceIter) || this.convergenceIter < 1) {
      throw new Error(`convergenceIter must be an integer >= 1. Got ${this.convergenceIter}.`);
    }
    if (this.preference !== undefined && !Number.isFinite(this.preference)) {
      throw new Error(`preference must be finite when provided. Got ${this.preference}.`);
    }
    if (this.randomState !== undefined && !Number.isFinite(this.randomState)) {
      throw new Error(`randomState must be finite when provided. Got ${this.randomState}.`);
    }
  }

  fit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    const n = X.length;
    const similarity: Matrix = Array.from({ length: n }, () => new Array<number>(n).fill(0));
    const similarityValues: number[] = [];
    for (let i = 0; i < n; i += 1) {
      for (let k = 0; k < n; k += 1) {
        if (i === k) {
          continue;
        }
        const s = -squaredDistance(X[i], X[k]);
        similarity[i][k] = s;
        similarityValues.push(s);
      }
    }
    const preference = this.preference ?? median(similarityValues);
    for (let i = 0; i < n; i += 1) {
      similarity[i][i] = preference;
    }

    const responsibility: Matrix = Array.from({ length: n }, () => new Array<number>(n).fill(0));
    const availability: Matrix = Array.from({ length: n }, () => new Array<number>(n).fill(0));
    const exemplarHistory: boolean[][] = [];
    let iterations = 0;
    const jitter =
      this.randomState === undefined
        ? 0
        : (Math.sin(this.randomState) + 1) * 1e-12;

    for (let iter = 0; iter < this.maxIter; iter += 1) {
      iterations = iter + 1;
      const updatedResponsibility: Matrix = Array.from({ length: n }, () => new Array<number>(n).fill(0));
      for (let i = 0; i < n; i += 1) {
        let best = Number.NEGATIVE_INFINITY;
        let secondBest = Number.NEGATIVE_INFINITY;
        let bestIndex = -1;
        for (let k = 0; k < n; k += 1) {
          const value = availability[i][k] + similarity[i][k];
          if (value > best) {
            secondBest = best;
            best = value;
            bestIndex = k;
          } else if (value > secondBest) {
            secondBest = value;
          }
        }
        for (let k = 0; k < n; k += 1) {
          const maxOther = k === bestIndex ? secondBest : best;
          const raw = similarity[i][k] - maxOther + (k === i ? jitter : 0);
          updatedResponsibility[i][k] =
            this.damping * responsibility[i][k] + (1 - this.damping) * raw;
        }
      }

      const updatedAvailability: Matrix = Array.from({ length: n }, () => new Array<number>(n).fill(0));
      for (let k = 0; k < n; k += 1) {
        let sumPositive = 0;
        for (let i = 0; i < n; i += 1) {
          if (i === k) {
            continue;
          }
          sumPositive += Math.max(0, updatedResponsibility[i][k]);
        }

        for (let i = 0; i < n; i += 1) {
          let raw: number;
          if (i === k) {
            raw = sumPositive;
          } else {
            raw = Math.min(
              0,
              updatedResponsibility[k][k] + sumPositive - Math.max(0, updatedResponsibility[i][k]),
            );
          }
          updatedAvailability[i][k] =
            this.damping * availability[i][k] + (1 - this.damping) * raw;
        }
      }

      for (let i = 0; i < n; i += 1) {
        for (let k = 0; k < n; k += 1) {
          responsibility[i][k] = updatedResponsibility[i][k];
          availability[i][k] = updatedAvailability[i][k];
        }
      }

      const exemplars = new Array<boolean>(n).fill(false);
      for (let k = 0; k < n; k += 1) {
        exemplars[k] = availability[k][k] + responsibility[k][k] > 0;
      }
      exemplarHistory.push(exemplars);
      if (exemplarHistory.length > this.convergenceIter) {
        exemplarHistory.shift();
      }

      if (exemplarHistory.length === this.convergenceIter) {
        let stable = true;
        for (let k = 0; k < n && stable; k += 1) {
          const value = exemplarHistory[0][k];
          for (let h = 1; h < exemplarHistory.length; h += 1) {
            if (exemplarHistory[h][k] !== value) {
              stable = false;
              break;
            }
          }
        }
        if (stable) {
          break;
        }
      }
    }

    let exemplars = new Array<number>();
    for (let k = 0; k < n; k += 1) {
      if (availability[k][k] + responsibility[k][k] > 0) {
        exemplars.push(k);
      }
    }
    if (exemplars.length === 0) {
      const score = new Array<number>(n);
      for (let k = 0; k < n; k += 1) {
        score[k] = availability[k][k] + responsibility[k][k];
      }
      exemplars = [argmax(score)];
    }

    const labels = new Array<number>(n);
    for (let i = 0; i < n; i += 1) {
      const similaritiesToExemplars = exemplars.map((index) => similarity[i][index]);
      labels[i] = argmax(similaritiesToExemplars);
    }

    this.clusterCentersIndices_ = exemplars.slice();
    this.clusterCenters_ = exemplars.map((index) => X[index].slice());
    this.labels_ = labels;
    this.nIter_ = iterations;
    this.nFeaturesIn_ = X[0].length;
    this.fitted = true;
    return this;
  }

  fitPredict(X: Matrix): Vector {
    return this.fit(X).labels_!.slice();
  }

  predict(X: Matrix): Vector {
    this.assertFitted();
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }

    const out = new Array<number>(X.length).fill(0);
    for (let i = 0; i < X.length; i += 1) {
      let bestIndex = 0;
      let bestDistance = Number.POSITIVE_INFINITY;
      for (let c = 0; c < this.clusterCenters_!.length; c += 1) {
        const d = squaredDistance(X[i], this.clusterCenters_![c]);
        if (d < bestDistance) {
          bestDistance = d;
          bestIndex = c;
        }
      }
      out[i] = bestIndex;
    }
    return out;
  }

  private assertFitted(): void {
    if (!this.fitted || !this.clusterCenters_ || this.nFeaturesIn_ === null) {
      throw new Error("AffinityPropagation has not been fitted.");
    }
  }
}
