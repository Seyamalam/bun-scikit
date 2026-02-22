import type { Matrix } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";

export interface PolynomialFeaturesOptions {
  degree?: number;
  includeBias?: boolean;
  interactionOnly?: boolean;
}

type TermPower = number[];

function generateCombinationsWithReplacement(
  nFeatures: number,
  degree: number,
  start: number,
  current: number[],
  out: number[][],
): void {
  if (current.length === degree) {
    out.push(current.slice());
    return;
  }

  for (let featureIndex = start; featureIndex < nFeatures; featureIndex += 1) {
    current.push(featureIndex);
    generateCombinationsWithReplacement(
      nFeatures,
      degree,
      featureIndex,
      current,
      out,
    );
    current.pop();
  }
}

function generateCombinationsWithoutReplacement(
  nFeatures: number,
  degree: number,
  start: number,
  current: number[],
  out: number[][],
): void {
  if (current.length === degree) {
    out.push(current.slice());
    return;
  }

  for (let featureIndex = start; featureIndex < nFeatures; featureIndex += 1) {
    current.push(featureIndex);
    generateCombinationsWithoutReplacement(
      nFeatures,
      degree,
      featureIndex + 1,
      current,
      out,
    );
    current.pop();
  }
}

export class PolynomialFeatures {
  nFeaturesIn_: number | null = null;
  nOutputFeatures_: number | null = null;
  powers_: number[][] | null = null;

  private readonly degree: number;
  private readonly includeBias: boolean;
  private readonly interactionOnly: boolean;

  constructor(options: PolynomialFeaturesOptions = {}) {
    this.degree = options.degree ?? 2;
    this.includeBias = options.includeBias ?? true;
    this.interactionOnly = options.interactionOnly ?? false;

    if (!Number.isInteger(this.degree) || this.degree < 0) {
      throw new Error(`degree must be an integer >= 0. Got ${this.degree}.`);
    }
  }

  fit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    const nFeatures = X[0].length;
    const powers: TermPower[] = [];

    if (this.includeBias) {
      powers.push(new Array<number>(nFeatures).fill(0));
    }

    for (let d = 1; d <= this.degree; d += 1) {
      const combinations: number[][] = [];
      if (this.interactionOnly) {
        generateCombinationsWithoutReplacement(nFeatures, d, 0, [], combinations);
      } else {
        generateCombinationsWithReplacement(nFeatures, d, 0, [], combinations);
      }

      for (let i = 0; i < combinations.length; i += 1) {
        const power = new Array<number>(nFeatures).fill(0);
        for (let j = 0; j < combinations[i].length; j += 1) {
          power[combinations[i][j]] += 1;
        }
        powers.push(power);
      }
    }

    this.nFeaturesIn_ = nFeatures;
    this.nOutputFeatures_ = powers.length;
    this.powers_ = powers;
    return this;
  }

  transform(X: Matrix): Matrix {
    if (this.nFeaturesIn_ === null || this.powers_ === null || this.nOutputFeatures_ === null) {
      throw new Error("PolynomialFeatures has not been fitted.");
    }

    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }

    const transformed = new Array<number[]>(X.length);
    for (let i = 0; i < X.length; i += 1) {
      const row = X[i];
      const outRow = new Array<number>(this.nOutputFeatures_);
      for (let termIndex = 0; termIndex < this.powers_.length; termIndex += 1) {
        const power = this.powers_[termIndex];
        let value = 1;
        for (let featureIndex = 0; featureIndex < this.nFeaturesIn_; featureIndex += 1) {
          const exponent = power[featureIndex];
          if (exponent === 1) {
            value *= row[featureIndex];
          } else if (exponent > 1) {
            value *= row[featureIndex] ** exponent;
          }
        }
        outRow[termIndex] = value;
      }
      transformed[i] = outRow;
    }
    return transformed;
  }

  fitTransform(X: Matrix): Matrix {
    return this.fit(X).transform(X);
  }
}
