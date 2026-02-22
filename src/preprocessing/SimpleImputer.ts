import type { Matrix, Vector } from "../types";
import { assertConsistentRowSize, assertNonEmptyMatrix } from "../utils/validation";

export type ImputerStrategy = "mean" | "median" | "most_frequent" | "constant";

export interface SimpleImputerOptions {
  strategy?: ImputerStrategy;
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

function median(values: number[]): number {
  if (values.length === 0) {
    throw new Error("Cannot compute median of an empty array.");
  }
  const sorted = values.slice().sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) {
    return 0.5 * (sorted[middle - 1] + sorted[middle]);
  }
  return sorted[middle];
}

function mostFrequent(values: number[]): number {
  if (values.length === 0) {
    throw new Error("Cannot compute most frequent of an empty array.");
  }
  const counts = new Map<number, number>();
  for (let i = 0; i < values.length; i += 1) {
    counts.set(values[i], (counts.get(values[i]) ?? 0) + 1);
  }

  let bestValue = values[0];
  let bestCount = counts.get(bestValue)!;
  for (const [value, count] of counts.entries()) {
    if (count > bestCount || (count === bestCount && value < bestValue)) {
      bestValue = value;
      bestCount = count;
    }
  }
  return bestValue;
}

export class SimpleImputer {
  statistics_: Vector | null = null;
  private readonly strategy: ImputerStrategy;
  private readonly fillValue?: number;

  constructor(options: SimpleImputerOptions = {}) {
    this.strategy = options.strategy ?? "mean";
    this.fillValue = options.fillValue;

    if (this.strategy === "constant") {
      const value = this.fillValue ?? 0;
      if (!Number.isFinite(value)) {
        throw new Error(`fillValue must be finite for constant strategy. Got ${value}.`);
      }
    }
  }

  fit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteOrMissing(X);

    const nFeatures = X[0].length;
    const stats = new Array<number>(nFeatures);

    for (let featureIndex = 0; featureIndex < nFeatures; featureIndex += 1) {
      const values: number[] = [];
      for (let i = 0; i < X.length; i += 1) {
        const value = X[i][featureIndex];
        if (!isMissing(value)) {
          values.push(value);
        }
      }

      if (values.length === 0) {
        if (this.strategy === "constant") {
          stats[featureIndex] = this.fillValue ?? 0;
          continue;
        }
        throw new Error(
          `Feature at index ${featureIndex} has only missing values. Use strategy='constant' or provide observed values.`,
        );
      }

      switch (this.strategy) {
        case "mean": {
          let sum = 0;
          for (let i = 0; i < values.length; i += 1) {
            sum += values[i];
          }
          stats[featureIndex] = sum / values.length;
          break;
        }
        case "median":
          stats[featureIndex] = median(values);
          break;
        case "most_frequent":
          stats[featureIndex] = mostFrequent(values);
          break;
        case "constant":
          stats[featureIndex] = this.fillValue ?? 0;
          break;
      }
    }

    this.statistics_ = stats;
    return this;
  }

  transform(X: Matrix): Matrix {
    if (!this.statistics_) {
      throw new Error("SimpleImputer has not been fitted.");
    }

    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteOrMissing(X);

    if (X[0].length !== this.statistics_.length) {
      throw new Error(
        `Feature size mismatch. Expected ${this.statistics_.length}, got ${X[0].length}.`,
      );
    }

    return X.map((row) =>
      row.map((value, featureIndex) => (isMissing(value) ? this.statistics_![featureIndex] : value)),
    );
  }

  fitTransform(X: Matrix): Matrix {
    return this.fit(X).transform(X);
  }
}
