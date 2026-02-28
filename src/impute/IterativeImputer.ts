import type { Matrix, Vector } from "../types";
import {
  assertConsistentRowSize,
  assertNonEmptyMatrix,
} from "../utils/validation";
import { Ridge } from "../linear_model/Ridge";

export type IterativeImputerInitialStrategy = "mean" | "median" | "most_frequent" | "constant";

export interface IterativeImputerOptions {
  maxIter?: number;
  tolerance?: number;
  initialStrategy?: IterativeImputerInitialStrategy;
  fillValue?: number;
}

function isMissing(value: number): boolean {
  return Number.isNaN(value);
}

function assertFiniteOrMissing(X: Matrix, label = "X"): void {
  for (let i = 0; i < X.length; i += 1) {
    for (let j = 0; j < X[i].length; j += 1) {
      const value = X[i][j];
      if (!Number.isFinite(value) && !isMissing(value)) {
        throw new Error(`${label} contains non-finite non-missing value at [${i}, ${j}].`);
      }
    }
  }
}

function median(values: Vector): number {
  const sorted = values.slice().sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? 0.5 * (sorted[mid - 1] + sorted[mid])
    : sorted[mid];
}

function mostFrequent(values: Vector): number {
  const counts = new Map<number, number>();
  for (let i = 0; i < values.length; i += 1) {
    counts.set(values[i], (counts.get(values[i]) ?? 0) + 1);
  }
  let bestValue = values[0];
  let bestCount = counts.get(bestValue) ?? 0;
  for (const [value, count] of counts.entries()) {
    if (count > bestCount || (count === bestCount && value < bestValue)) {
      bestValue = value;
      bestCount = count;
    }
  }
  return bestValue;
}

function removeColumn(row: Vector, column: number): Vector {
  const out = new Array<number>(row.length - 1);
  let offset = 0;
  for (let i = 0; i < row.length; i += 1) {
    if (i === column) {
      continue;
    }
    out[offset] = row[i];
    offset += 1;
  }
  return out;
}

export class IterativeImputer {
  nFeaturesIn_: number | null = null;
  nIter_ = 0;
  imputationSequence_: Vector | null = null;

  private maxIter: number;
  private tolerance: number;
  private initialStrategy: IterativeImputerInitialStrategy;
  private fillValue?: number;
  private initialStatistics_: Vector | null = null;
  private models_: Map<number, Ridge> = new Map<number, Ridge>();
  private missingFeatures_: Set<number> = new Set<number>();
  private fitted = false;

  constructor(options: IterativeImputerOptions = {}) {
    this.maxIter = options.maxIter ?? 10;
    this.tolerance = options.tolerance ?? 1e-3;
    this.initialStrategy = options.initialStrategy ?? "mean";
    this.fillValue = options.fillValue;
    if (!Number.isInteger(this.maxIter) || this.maxIter < 1) {
      throw new Error(`maxIter must be an integer >= 1. Got ${this.maxIter}.`);
    }
    if (!Number.isFinite(this.tolerance) || this.tolerance < 0) {
      throw new Error(`tolerance must be finite and >= 0. Got ${this.tolerance}.`);
    }
  }

  fit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteOrMissing(X);

    const nSamples = X.length;
    const nFeatures = X[0].length;
    const missingMask = X.map((row) => row.map((value) => isMissing(value)));
    const statistics = this.computeInitialStatistics(X);
    let imputed = this.applyInitialImputation(X, statistics);

    const sequence: number[] = [];
    this.models_.clear();
    this.missingFeatures_.clear();

    for (let j = 0; j < nFeatures; j += 1) {
      let hasMissing = false;
      for (let i = 0; i < nSamples; i += 1) {
        if (missingMask[i][j]) {
          hasMissing = true;
          break;
        }
      }
      if (hasMissing) {
        sequence.push(j);
        this.missingFeatures_.add(j);
      }
    }

    let nIter = 0;
    for (let iter = 0; iter < this.maxIter; iter += 1) {
      nIter = iter + 1;
      const previous = imputed.map((row) => row.slice());

      for (let s = 0; s < sequence.length; s += 1) {
        const targetFeature = sequence[s];
        const observedIndices: number[] = [];
        const missingIndices: number[] = [];
        for (let i = 0; i < nSamples; i += 1) {
          if (missingMask[i][targetFeature]) {
            missingIndices.push(i);
          } else {
            observedIndices.push(i);
          }
        }
        if (missingIndices.length === 0 || observedIndices.length < 2) {
          continue;
        }

        const XTrain = observedIndices.map((idx) => removeColumn(imputed[idx], targetFeature));
        const yTrain = observedIndices.map((idx) => imputed[idx][targetFeature]);
        const XMissing = missingIndices.map((idx) => removeColumn(imputed[idx], targetFeature));

        const model = new Ridge({ alpha: 1, fitIntercept: true }).fit(XTrain, yTrain);
        const pred = model.predict(XMissing);
        for (let i = 0; i < missingIndices.length; i += 1) {
          imputed[missingIndices[i]][targetFeature] = pred[i];
        }
        this.models_.set(targetFeature, model);
      }

      let maxDelta = 0;
      for (let i = 0; i < nSamples; i += 1) {
        for (let j = 0; j < nFeatures; j += 1) {
          if (!missingMask[i][j]) {
            continue;
          }
          const delta = Math.abs(imputed[i][j] - previous[i][j]);
          if (delta > maxDelta) {
            maxDelta = delta;
          }
        }
      }

      if (maxDelta < this.tolerance) {
        break;
      }
    }

    this.nFeaturesIn_ = nFeatures;
    this.nIter_ = nIter;
    this.imputationSequence_ = sequence;
    this.initialStatistics_ = statistics;
    this.fitted = true;
    return this;
  }

  transform(X: Matrix): Matrix {
    this.assertFitted();
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteOrMissing(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }

    const missingMask = X.map((row) => row.map((value) => isMissing(value)));
    const imputed = this.applyInitialImputation(X, this.initialStatistics_!);

    for (let s = 0; s < this.imputationSequence_!.length; s += 1) {
      const feature = this.imputationSequence_![s];
      const model = this.models_.get(feature);
      if (!model) {
        continue;
      }
      const missingIndices: number[] = [];
      for (let i = 0; i < imputed.length; i += 1) {
        if (missingMask[i][feature]) {
          missingIndices.push(i);
        }
      }
      if (missingIndices.length === 0) {
        continue;
      }
      const XMissing = missingIndices.map((idx) => removeColumn(imputed[idx], feature));
      const pred = model.predict(XMissing);
      for (let i = 0; i < missingIndices.length; i += 1) {
        imputed[missingIndices[i]][feature] = pred[i];
      }
    }

    return imputed;
  }

  fitTransform(X: Matrix): Matrix {
    return this.fit(X).transform(X);
  }

  private computeInitialStatistics(X: Matrix): Vector {
    const nFeatures = X[0].length;
    const stats = new Array<number>(nFeatures).fill(0);
    for (let j = 0; j < nFeatures; j += 1) {
      const observed: number[] = [];
      for (let i = 0; i < X.length; i += 1) {
        if (!isMissing(X[i][j])) {
          observed.push(X[i][j]);
        }
      }
      if (observed.length === 0) {
        stats[j] = this.fillValue ?? 0;
        continue;
      }
      switch (this.initialStrategy) {
        case "mean": {
          let sum = 0;
          for (let i = 0; i < observed.length; i += 1) {
            sum += observed[i];
          }
          stats[j] = sum / observed.length;
          break;
        }
        case "median":
          stats[j] = median(observed);
          break;
        case "most_frequent":
          stats[j] = mostFrequent(observed);
          break;
        case "constant":
          stats[j] = this.fillValue ?? 0;
          break;
      }
    }
    return stats;
  }

  private applyInitialImputation(X: Matrix, stats: Vector): Matrix {
    return X.map((row) =>
      row.map((value, featureIndex) => (isMissing(value) ? stats[featureIndex] : value)),
    );
  }

  private assertFitted(): void {
    if (!this.fitted || !this.imputationSequence_ || !this.initialStatistics_ || this.nFeaturesIn_ === null) {
      throw new Error("IterativeImputer has not been fitted.");
    }
  }
}
