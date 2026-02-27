import type { Matrix, Vector } from "../types";
import { accuracyScore } from "../metrics/classification";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  assertNonEmptyMatrix,
  assertVectorLength,
} from "../utils/validation";

export type DummyClassifierStrategy =
  | "most_frequent"
  | "prior"
  | "stratified"
  | "uniform"
  | "constant";

export interface DummyClassifierOptions {
  strategy?: DummyClassifierStrategy;
  constant?: number;
  randomState?: number;
}

class Mulberry32 {
  private state: number;

  constructor(seed: number) {
    this.state = seed >>> 0;
  }

  next(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let t = this.state ^ (this.state >>> 15);
    t = Math.imul(t, this.state | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
}

export class DummyClassifier {
  classes_: number[] | null = null;
  classPrior_: number[] | null = null;
  constant_: number | null = null;

  private readonly strategy: DummyClassifierStrategy;
  private readonly configuredConstant?: number;
  private readonly randomState: number;
  private majorityClass: number | null = null;
  private nFeaturesIn_: number | null = null;

  constructor(options: DummyClassifierOptions = {}) {
    this.strategy = options.strategy ?? "prior";
    this.configuredConstant = options.constant;
    this.randomState = options.randomState ?? 42;
  }

  fit(X: Matrix, y: Vector, sampleWeight?: Vector): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    assertVectorLength(y, X.length);
    assertFiniteVector(y);
    this.nFeaturesIn_ = X[0].length;

    const counts = new Map<number, number>();
    for (let i = 0; i < y.length; i += 1) {
      counts.set(y[i], (counts.get(y[i]) ?? 0) + 1);
    }

    const classes = Array.from(counts.keys()).sort((a, b) => a - b);
    const priors = new Array<number>(classes.length);
    for (let i = 0; i < classes.length; i += 1) {
      priors[i] = (counts.get(classes[i]) ?? 0) / y.length;
    }

    let majorityClass = classes[0];
    let majorityCount = counts.get(majorityClass) ?? 0;
    for (let i = 1; i < classes.length; i += 1) {
      const cls = classes[i];
      const clsCount = counts.get(cls) ?? 0;
      if (clsCount > majorityCount) {
        majorityClass = cls;
        majorityCount = clsCount;
      }
    }

    if (this.strategy === "constant") {
      if (!Number.isFinite(this.configuredConstant)) {
        throw new Error("constant strategy requires a finite constant value.");
      }
      this.constant_ = this.configuredConstant!;
    } else {
      this.constant_ = majorityClass;
    }

    this.classes_ = classes;
    this.classPrior_ = priors;
    this.majorityClass = majorityClass;
    return this;
  }

  private ensureFitted(): void {
    if (!this.classes_ || !this.classPrior_ || this.nFeaturesIn_ === null || this.majorityClass === null) {
      throw new Error("DummyClassifier has not been fitted.");
    }
  }

  private sampleByPrior(rng: Mulberry32): number {
    let r = rng.next();
    for (let i = 0; i < this.classPrior_!.length; i += 1) {
      r -= this.classPrior_![i];
      if (r <= 0) {
        return this.classes_![i];
      }
    }
    return this.classes_![this.classes_!.length - 1];
  }

  predict(X: Matrix): Vector {
    this.ensureFitted();
    if (!Array.isArray(X) || X.length === 0) {
      throw new Error("X must be a non-empty 2D array.");
    }
    if (!Array.isArray(X[0]) || X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0]?.length ?? 0}.`);
    }

    switch (this.strategy) {
      case "most_frequent":
      case "prior":
        return new Array<number>(X.length).fill(this.majorityClass!);
      case "constant":
        return new Array<number>(X.length).fill(this.constant_!);
      case "uniform": {
        const rng = new Mulberry32(this.randomState);
        const out = new Array<number>(X.length);
        for (let i = 0; i < X.length; i += 1) {
          const idx = Math.floor(rng.next() * this.classes_!.length);
          out[i] = this.classes_![idx];
        }
        return out;
      }
      case "stratified": {
        const rng = new Mulberry32(this.randomState);
        const out = new Array<number>(X.length);
        for (let i = 0; i < X.length; i += 1) {
          out[i] = this.sampleByPrior(rng);
        }
        return out;
      }
      default: {
        const exhaustive: never = this.strategy;
        throw new Error(`Unsupported strategy: ${exhaustive}`);
      }
    }
  }

  predictProba(X: Matrix): Matrix {
    this.ensureFitted();
    if (!Array.isArray(X) || X.length === 0) {
      throw new Error("X must be a non-empty 2D array.");
    }
    if (!Array.isArray(X[0]) || X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0]?.length ?? 0}.`);
    }

    if (this.strategy === "uniform") {
      const value = 1 / this.classes_!.length;
      return X.map(() => new Array(this.classes_!.length).fill(value));
    }

    if (this.strategy === "most_frequent" || this.strategy === "constant") {
      const oneHot = new Array<number>(this.classes_!.length).fill(0);
      const label = this.strategy === "constant" ? this.constant_! : this.majorityClass!;
      const classIndex = this.classes_!.indexOf(label);
      if (classIndex >= 0) {
        oneHot[classIndex] = 1;
      }
      return X.map(() => [...oneHot]);
    }

    // prior / stratified share prior probabilities.
    const prior = [...this.classPrior_!];
    return X.map(() => [...prior]);
  }

  score(X: Matrix, y: Vector): number {
    return accuracyScore(y, this.predict(X));
  }
}

