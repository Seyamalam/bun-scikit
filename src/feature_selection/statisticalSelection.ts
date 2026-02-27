import type { Matrix, Vector } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  assertNonEmptyMatrix,
  assertVectorLength,
} from "../utils/validation";
import {
  SelectKBest,
  SelectPercentile,
  f_classif,
  type UnivariateScoreFunc,
} from "./univariateSelection";
import { crossValScore, type BuiltInScoring, type CrossValSplitter, type ScoringFn } from "../model_selection/crossValScore";

function validateXy(X: Matrix, y: Vector): void {
  assertNonEmptyMatrix(X);
  assertConsistentRowSize(X);
  assertFiniteMatrix(X);
  assertVectorLength(y, X.length);
  assertFiniteVector(y);
}

function supportMaskToIndices(mask: boolean[]): number[] {
  const out: number[] = [];
  for (let i = 0; i < mask.length; i += 1) {
    if (mask[i]) {
      out.push(i);
    }
  }
  return out;
}

function applySupportMask(X: Matrix, mask: boolean[]): Matrix {
  const indices = supportMaskToIndices(mask);
  return X.map((row) => indices.map((index) => row[index]));
}

function ensureAtLeastOne(mask: boolean[], pValues: Vector): void {
  if (mask.some(Boolean)) {
    return;
  }
  let bestIndex = 0;
  let bestValue = Number.POSITIVE_INFINITY;
  for (let i = 0; i < pValues.length; i += 1) {
    if (pValues[i] < bestValue) {
      bestValue = pValues[i];
      bestIndex = i;
    }
  }
  mask[bestIndex] = true;
}

export interface SelectFprOptions {
  scoreFunc?: UnivariateScoreFunc;
  alpha?: number;
}

export class SelectFpr {
  scores_: Vector | null = null;
  pvalues_: Vector | null = null;
  support_: boolean[] | null = null;
  nFeaturesIn_: number | null = null;

  private scoreFunc: UnivariateScoreFunc;
  private alpha: number;

  constructor(options: SelectFprOptions = {}) {
    this.scoreFunc = options.scoreFunc ?? f_classif;
    this.alpha = options.alpha ?? 0.05;
    if (!Number.isFinite(this.alpha) || this.alpha <= 0 || this.alpha >= 1) {
      throw new Error(`alpha must be in (0, 1). Got ${this.alpha}.`);
    }
  }

  fit(X: Matrix, y: Vector): this {
    validateXy(X, y);
    const [scores, pValues] = this.scoreFunc(X, y);
    const support = pValues.map((value) => value <= this.alpha);
    ensureAtLeastOne(support, pValues);
    this.scores_ = scores.slice();
    this.pvalues_ = pValues.slice();
    this.support_ = support;
    this.nFeaturesIn_ = X[0].length;
    return this;
  }

  transform(X: Matrix): Matrix {
    if (!this.support_ || this.nFeaturesIn_ === null) {
      throw new Error("SelectFpr has not been fitted.");
    }
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }
    return applySupportMask(X, this.support_);
  }

  fitTransform(X: Matrix, y: Vector): Matrix {
    return this.fit(X, y).transform(X);
  }

  getSupport(indices = false): boolean[] | number[] {
    if (!this.support_) {
      throw new Error("SelectFpr has not been fitted.");
    }
    return indices ? supportMaskToIndices(this.support_) : this.support_.slice();
  }
}

export interface SelectFdrOptions {
  scoreFunc?: UnivariateScoreFunc;
  alpha?: number;
}

export class SelectFdr {
  scores_: Vector | null = null;
  pvalues_: Vector | null = null;
  support_: boolean[] | null = null;
  nFeaturesIn_: number | null = null;

  private scoreFunc: UnivariateScoreFunc;
  private alpha: number;

  constructor(options: SelectFdrOptions = {}) {
    this.scoreFunc = options.scoreFunc ?? f_classif;
    this.alpha = options.alpha ?? 0.05;
    if (!Number.isFinite(this.alpha) || this.alpha <= 0 || this.alpha >= 1) {
      throw new Error(`alpha must be in (0, 1). Got ${this.alpha}.`);
    }
  }

  fit(X: Matrix, y: Vector): this {
    validateXy(X, y);
    const [scores, pValues] = this.scoreFunc(X, y);
    const m = pValues.length;
    const order = Array.from({ length: m }, (_, i) => i).sort((a, b) => pValues[a] - pValues[b]);
    let cutoff = Number.NEGATIVE_INFINITY;
    for (let rank = 1; rank <= m; rank += 1) {
      const p = pValues[order[rank - 1]];
      const threshold = (this.alpha * rank) / m;
      if (p <= threshold) {
        cutoff = p;
      }
    }
    const support = pValues.map((value) => value <= cutoff);
    ensureAtLeastOne(support, pValues);
    this.scores_ = scores.slice();
    this.pvalues_ = pValues.slice();
    this.support_ = support;
    this.nFeaturesIn_ = X[0].length;
    return this;
  }

  transform(X: Matrix): Matrix {
    if (!this.support_ || this.nFeaturesIn_ === null) {
      throw new Error("SelectFdr has not been fitted.");
    }
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }
    return applySupportMask(X, this.support_);
  }

  fitTransform(X: Matrix, y: Vector): Matrix {
    return this.fit(X, y).transform(X);
  }

  getSupport(indices = false): boolean[] | number[] {
    if (!this.support_) {
      throw new Error("SelectFdr has not been fitted.");
    }
    return indices ? supportMaskToIndices(this.support_) : this.support_.slice();
  }
}

export interface SelectFweOptions {
  scoreFunc?: UnivariateScoreFunc;
  alpha?: number;
}

export class SelectFwe {
  scores_: Vector | null = null;
  pvalues_: Vector | null = null;
  support_: boolean[] | null = null;
  nFeaturesIn_: number | null = null;

  private scoreFunc: UnivariateScoreFunc;
  private alpha: number;

  constructor(options: SelectFweOptions = {}) {
    this.scoreFunc = options.scoreFunc ?? f_classif;
    this.alpha = options.alpha ?? 0.05;
    if (!Number.isFinite(this.alpha) || this.alpha <= 0 || this.alpha >= 1) {
      throw new Error(`alpha must be in (0, 1). Got ${this.alpha}.`);
    }
  }

  fit(X: Matrix, y: Vector): this {
    validateXy(X, y);
    const [scores, pValues] = this.scoreFunc(X, y);
    const threshold = this.alpha / Math.max(1, pValues.length);
    const support = pValues.map((value) => value <= threshold);
    ensureAtLeastOne(support, pValues);
    this.scores_ = scores.slice();
    this.pvalues_ = pValues.slice();
    this.support_ = support;
    this.nFeaturesIn_ = X[0].length;
    return this;
  }

  transform(X: Matrix): Matrix {
    if (!this.support_ || this.nFeaturesIn_ === null) {
      throw new Error("SelectFwe has not been fitted.");
    }
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }
    return applySupportMask(X, this.support_);
  }

  fitTransform(X: Matrix, y: Vector): Matrix {
    return this.fit(X, y).transform(X);
  }

  getSupport(indices = false): boolean[] | number[] {
    if (!this.support_) {
      throw new Error("SelectFwe has not been fitted.");
    }
    return indices ? supportMaskToIndices(this.support_) : this.support_.slice();
  }
}

export type GenericUnivariateSelectMode = "percentile" | "k_best" | "fpr" | "fdr" | "fwe";

export interface GenericUnivariateSelectOptions {
  scoreFunc?: UnivariateScoreFunc;
  mode?: GenericUnivariateSelectMode;
  param?: number;
}

type GenericSelector =
  | SelectPercentile
  | SelectKBest
  | SelectFpr
  | SelectFdr
  | SelectFwe;

export class GenericUnivariateSelect {
  scores_: Vector | null = null;
  pvalues_: Vector | null = null;
  nFeaturesIn_: number | null = null;

  private scoreFunc: UnivariateScoreFunc;
  private mode: GenericUnivariateSelectMode;
  private param: number;
  private selector: GenericSelector | null = null;

  constructor(options: GenericUnivariateSelectOptions = {}) {
    this.scoreFunc = options.scoreFunc ?? f_classif;
    this.mode = options.mode ?? "percentile";
    this.param = options.param ?? (this.mode === "percentile" ? 10 : this.mode === "k_best" ? 10 : 0.05);
  }

  fit(X: Matrix, y: Vector): this {
    validateXy(X, y);
    switch (this.mode) {
      case "percentile":
        this.selector = new SelectPercentile({ scoreFunc: this.scoreFunc, percentile: this.param });
        break;
      case "k_best":
        this.selector = new SelectKBest({ scoreFunc: this.scoreFunc, k: this.param });
        break;
      case "fpr":
        this.selector = new SelectFpr({ scoreFunc: this.scoreFunc, alpha: this.param });
        break;
      case "fdr":
        this.selector = new SelectFdr({ scoreFunc: this.scoreFunc, alpha: this.param });
        break;
      case "fwe":
        this.selector = new SelectFwe({ scoreFunc: this.scoreFunc, alpha: this.param });
        break;
      default: {
        const exhaustive: never = this.mode;
        throw new Error(`Unsupported mode: ${exhaustive}`);
      }
    }

    this.selector.fit(X, y);
    this.scores_ = (this.selector as { scores_?: Vector | null }).scores_?.slice() ?? null;
    this.pvalues_ = (this.selector as { pvalues_?: Vector | null }).pvalues_?.slice() ?? null;
    this.nFeaturesIn_ = X[0].length;
    return this;
  }

  transform(X: Matrix): Matrix {
    if (!this.selector) {
      throw new Error("GenericUnivariateSelect has not been fitted.");
    }
    return this.selector.transform(X);
  }

  fitTransform(X: Matrix, y: Vector): Matrix {
    return this.fit(X, y).transform(X);
  }

  getSupport(indices = false): boolean[] | number[] {
    if (!this.selector) {
      throw new Error("GenericUnivariateSelect has not been fitted.");
    }
    return this.selector.getSupport(indices);
  }
}

type EstimatorLike = {
  fit(X: Matrix, y: Vector, sampleWeight?: Vector): unknown;
  predict(X: Matrix): Vector;
  score?(X: Matrix, y: Vector): number;
  getParams?(deep?: boolean): unknown;
};

function cloneEstimatorFromPrototype(estimator: EstimatorLike): EstimatorLike {
  const ctor = (estimator as unknown as { constructor?: new (args?: unknown) => EstimatorLike }).constructor;
  if (typeof ctor !== "function") {
    return estimator;
  }
  const params = typeof estimator.getParams === "function" ? estimator.getParams(true) : undefined;
  try {
    if (params && typeof params === "object" && Object.keys(params).length > 0) {
      return new ctor(params);
    }
    return new ctor();
  } catch {
    return estimator;
  }
}

function resolveEstimator(input: (() => EstimatorLike) | EstimatorLike): EstimatorLike {
  if (typeof input === "function") {
    return input();
  }
  return cloneEstimatorFromPrototype(input);
}

function mean(values: number[]): number {
  let sum = 0;
  for (let i = 0; i < values.length; i += 1) {
    sum += values[i];
  }
  return values.length === 0 ? 0 : sum / values.length;
}

export type SequentialFeatureSelectorDirection = "forward" | "backward";

export interface SequentialFeatureSelectorOptions {
  nFeaturesToSelect?: number;
  direction?: SequentialFeatureSelectorDirection;
  scoring?: BuiltInScoring | ScoringFn;
  cv?: number | CrossValSplitter;
}

export class SequentialFeatureSelector {
  support_: boolean[] | null = null;
  selectedFeatureIndices_: number[] | null = null;
  nFeaturesIn_: number | null = null;

  private estimatorFactory: (() => EstimatorLike) | EstimatorLike;
  private nFeaturesToSelect?: number;
  private direction: SequentialFeatureSelectorDirection;
  private scoring?: BuiltInScoring | ScoringFn;
  private cv?: number | CrossValSplitter;

  constructor(
    estimatorFactory: (() => EstimatorLike) | EstimatorLike,
    options: SequentialFeatureSelectorOptions = {},
  ) {
    this.estimatorFactory = estimatorFactory;
    this.nFeaturesToSelect = options.nFeaturesToSelect;
    this.direction = options.direction ?? "forward";
    this.scoring = options.scoring;
    this.cv = options.cv;
  }

  fit(X: Matrix, y: Vector, sampleWeight?: Vector): this {
    validateXy(X, y);
    if (sampleWeight) {
      assertVectorLength(sampleWeight, X.length, "sampleWeight");
      assertFiniteVector(sampleWeight, "sampleWeight");
    }

    const nFeatures = X[0].length;
    const nSelect = this.nFeaturesToSelect ?? Math.max(1, Math.floor(nFeatures / 2));
    if (!Number.isInteger(nSelect) || nSelect < 1 || nSelect > nFeatures) {
      throw new Error(`nFeaturesToSelect must be integer in [1, ${nFeatures}]. Got ${nSelect}.`);
    }

    let selected =
      this.direction === "forward"
        ? ([] as number[])
        : Array.from({ length: nFeatures }, (_, idx) => idx);

    const targetCount = nSelect;
    while (
      (this.direction === "forward" && selected.length < targetCount) ||
      (this.direction === "backward" && selected.length > targetCount)
    ) {
      let bestScore = Number.NEGATIVE_INFINITY;
      let bestCandidate = -1;

      const candidates =
        this.direction === "forward"
          ? Array.from({ length: nFeatures }, (_, idx) => idx).filter((idx) => !selected.includes(idx))
          : selected.slice();

      for (let i = 0; i < candidates.length; i += 1) {
        const candidate = candidates[i];
        const trial =
          this.direction === "forward"
            ? selected.concat(candidate).sort((a, b) => a - b)
            : selected.filter((idx) => idx !== candidate);
        const XTrial = X.map((row) => trial.map((idx) => row[idx]));
        const scores = crossValScore(
          () => resolveEstimator(this.estimatorFactory),
          XTrial,
          y,
          {
            cv: this.cv,
            scoring: this.scoring,
            sampleWeight,
          },
        );
        const score = mean(scores);
        if (score > bestScore) {
          bestScore = score;
          bestCandidate = candidate;
        }
      }

      if (bestCandidate < 0) {
        break;
      }
      if (this.direction === "forward") {
        selected.push(bestCandidate);
        selected.sort((a, b) => a - b);
      } else {
        selected = selected.filter((idx) => idx !== bestCandidate);
      }
    }

    const support = new Array<boolean>(nFeatures).fill(false);
    for (let i = 0; i < selected.length; i += 1) {
      support[selected[i]] = true;
    }
    this.support_ = support;
    this.selectedFeatureIndices_ = selected.slice();
    this.nFeaturesIn_ = nFeatures;
    return this;
  }

  transform(X: Matrix): Matrix {
    if (!this.support_ || !this.selectedFeatureIndices_ || this.nFeaturesIn_ === null) {
      throw new Error("SequentialFeatureSelector has not been fitted.");
    }
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }
    return X.map((row) => this.selectedFeatureIndices_!.map((idx) => row[idx]));
  }

  fitTransform(X: Matrix, y: Vector, sampleWeight?: Vector): Matrix {
    return this.fit(X, y, sampleWeight).transform(X);
  }

  getSupport(indices = false): boolean[] | number[] {
    if (!this.support_ || !this.selectedFeatureIndices_) {
      throw new Error("SequentialFeatureSelector has not been fitted.");
    }
    if (indices) {
      return this.selectedFeatureIndices_.slice();
    }
    return this.support_.slice();
  }
}
