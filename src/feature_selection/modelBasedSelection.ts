import type { Matrix, Vector } from "../types";
import { crossValScore, type BuiltInScoring, type CrossValSplitter, type ScoringFn } from "../model_selection/crossValScore";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  assertNonEmptyMatrix,
  assertVectorLength,
} from "../utils/validation";

type ImportanceEstimatorLike = {
  fit(X: Matrix, y: Vector, sampleWeight?: Vector): unknown;
  predict(X: Matrix): Vector;
  score?(X: Matrix, y: Vector): number;
  getParams?: (deep?: boolean) => unknown;
  featureImportances_?: Vector | null;
  coef_?: Vector | Matrix | null;
};

export interface MutualInfoOptions {
  nBins?: number;
}

export interface SelectFromModelOptions {
  threshold?: number | "mean" | "median";
  maxFeatures?: number;
  prefit?: boolean;
  importanceGetter?: (estimator: ImportanceEstimatorLike) => Vector;
}

export interface RFEOptions {
  nFeaturesToSelect?: number;
  step?: number;
  importanceGetter?: (estimator: ImportanceEstimatorLike) => Vector;
}

export interface RFECVOptions extends RFEOptions {
  cv?: number | CrossValSplitter;
  scoring?: BuiltInScoring | ScoringFn;
  minFeaturesToSelect?: number;
}

function validateXy(X: Matrix, y: Vector): void {
  assertNonEmptyMatrix(X);
  assertConsistentRowSize(X);
  assertFiniteMatrix(X);
  assertVectorLength(y, X.length);
  assertFiniteVector(y);
}

function mean(values: Vector): number {
  if (values.length === 0) {
    return 0;
  }
  let total = 0;
  for (let i = 0; i < values.length; i += 1) {
    total += values[i];
  }
  return total / values.length;
}

function median(values: Vector): number {
  if (values.length === 0) {
    return 0;
  }
  const sorted = values.slice().sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) {
    return (sorted[middle - 1] + sorted[middle]) / 2;
  }
  return sorted[middle];
}

function rankFeatureIndices(scores: Vector, take: number, descending: boolean): number[] {
  const order = Array.from({ length: scores.length }, (_, index) => index);
  order.sort((a, b) => {
    const scoreA = Number.isFinite(scores[a]) ? scores[a] : Number.NEGATIVE_INFINITY;
    const scoreB = Number.isFinite(scores[b]) ? scores[b] : Number.NEGATIVE_INFINITY;
    if (scoreB !== scoreA) {
      return descending ? scoreB - scoreA : scoreA - scoreB;
    }
    return a - b;
  });
  const k = Math.max(0, Math.min(take, scores.length));
  return order.slice(0, k);
}

function subsetColumns(X: Matrix, selected: number[]): Matrix {
  return X.map((row) => selected.map((featureIndex) => row[featureIndex]));
}

function estimateThreshold(importances: Vector, threshold: number | "mean" | "median"): number {
  if (typeof threshold === "number") {
    return threshold;
  }
  if (threshold === "mean") {
    return mean(importances);
  }
  return median(importances);
}

function cloneEstimatorFromPrototype(
  estimator: ImportanceEstimatorLike,
): ImportanceEstimatorLike {
  const ctor = (estimator as unknown as { constructor?: new (args?: unknown) => ImportanceEstimatorLike }).constructor;
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

function resolveEstimator(
  input: (() => ImportanceEstimatorLike) | ImportanceEstimatorLike,
): ImportanceEstimatorLike {
  if (typeof input === "function") {
    return input();
  }
  return cloneEstimatorFromPrototype(input);
}

function normalizeImportances(values: Vector): Vector {
  return values.map((value) => Math.abs(Number.isFinite(value) ? value : 0));
}

function extractImportances(
  estimator: ImportanceEstimatorLike,
  importanceGetter?: (estimator: ImportanceEstimatorLike) => Vector,
): Vector {
  if (importanceGetter) {
    return normalizeImportances(importanceGetter(estimator));
  }
  if (Array.isArray(estimator.featureImportances_)) {
    return normalizeImportances(estimator.featureImportances_);
  }
  const coef = estimator.coef_;
  if (Array.isArray(coef)) {
    if (coef.length === 0) {
      throw new Error("Estimator coefficients are empty.");
    }
    if (Array.isArray(coef[0])) {
      const matrix = coef as Matrix;
      const out = new Array<number>(matrix[0].length).fill(0);
      for (let rowIndex = 0; rowIndex < matrix.length; rowIndex += 1) {
        for (let colIndex = 0; colIndex < matrix[rowIndex].length; colIndex += 1) {
          out[colIndex] += Math.abs(matrix[rowIndex][colIndex]);
        }
      }
      return out.map((value) => value / matrix.length);
    }
    return normalizeImportances(coef as Vector);
  }
  throw new Error(
    "Estimator does not expose feature importances via featureImportances_ or coef_.",
  );
}

function resolveNFeaturesToSelect(
  requested: number | undefined,
  totalFeatures: number,
): number {
  if (requested === undefined) {
    return Math.max(1, Math.floor(totalFeatures / 2));
  }
  if (!Number.isFinite(requested)) {
    throw new Error(`nFeaturesToSelect must be finite. Got ${requested}.`);
  }
  if (Number.isInteger(requested) && requested >= 1 && requested <= totalFeatures) {
    return requested;
  }
  if (requested > 0 && requested < 1) {
    return Math.max(1, Math.floor(totalFeatures * requested));
  }
  throw new Error(
    `nFeaturesToSelect must be in (0, 1] or integer in [1, ${totalFeatures}]. Got ${requested}.`,
  );
}

function resolveStep(step: number, currentFeatureCount: number): number {
  if (!Number.isFinite(step) || step <= 0) {
    throw new Error(`step must be finite and > 0. Got ${step}.`);
  }
  if (step < 1) {
    return Math.max(1, Math.floor(currentFeatureCount * step));
  }
  if (!Number.isInteger(step)) {
    throw new Error(`step >= 1 must be an integer. Got ${step}.`);
  }
  return step;
}

function discretize(values: Vector, nBins: number): number[] {
  if (nBins < 2) {
    return values.map(() => 0);
  }
  const sorted = values.slice().sort((a, b) => a - b);
  const cutPoints = new Array<number>(nBins - 1);
  for (let i = 1; i < nBins; i += 1) {
    const pos = (i * (sorted.length - 1)) / nBins;
    const low = Math.floor(pos);
    const high = Math.ceil(pos);
    const alpha = pos - low;
    cutPoints[i - 1] = sorted[low] * (1 - alpha) + sorted[high] * alpha;
  }

  return values.map((value) => {
    let bin = 0;
    while (bin < cutPoints.length && value > cutPoints[bin]) {
      bin += 1;
    }
    return bin;
  });
}

function mutualInformationFromDiscretePairs(xBins: number[], yBins: number[]): number {
  const n = xBins.length;
  const joint = new Map<string, number>();
  const xCounts = new Map<number, number>();
  const yCounts = new Map<number, number>();

  for (let i = 0; i < n; i += 1) {
    const x = xBins[i];
    const y = yBins[i];
    const key = `${x}|${y}`;
    joint.set(key, (joint.get(key) ?? 0) + 1);
    xCounts.set(x, (xCounts.get(x) ?? 0) + 1);
    yCounts.set(y, (yCounts.get(y) ?? 0) + 1);
  }

  let mi = 0;
  for (const [key, count] of joint.entries()) {
    const [xLabelRaw, yLabelRaw] = key.split("|");
    const xLabel = Number(xLabelRaw);
    const yLabel = Number(yLabelRaw);
    const pxy = count / n;
    const px = (xCounts.get(xLabel) ?? 0) / n;
    const py = (yCounts.get(yLabel) ?? 0) / n;
    if (pxy > 0 && px > 0 && py > 0) {
      mi += pxy * Math.log(pxy / (px * py));
    }
  }
  return Math.max(0, mi);
}

export function mutualInfoClassif(
  X: Matrix,
  y: Vector,
  options: MutualInfoOptions = {},
): Vector {
  validateXy(X, y);
  const nBins = options.nBins ?? 10;
  if (!Number.isInteger(nBins) || nBins < 2) {
    throw new Error(`nBins must be an integer >= 2. Got ${nBins}.`);
  }
  const classMap = new Map<number, number>();
  let nextClassIndex = 0;
  const yBins = y.map((label) => {
    const existing = classMap.get(label);
    if (existing !== undefined) {
      return existing;
    }
    classMap.set(label, nextClassIndex);
    nextClassIndex += 1;
    return nextClassIndex - 1;
  });

  const nFeatures = X[0].length;
  const scores = new Array<number>(nFeatures).fill(0);
  for (let feature = 0; feature < nFeatures; feature += 1) {
    const xColumn = X.map((row) => row[feature]);
    const xBins = discretize(xColumn, nBins);
    scores[feature] = mutualInformationFromDiscretePairs(xBins, yBins);
  }
  return scores;
}

export function mutualInfoRegression(
  X: Matrix,
  y: Vector,
  options: MutualInfoOptions = {},
): Vector {
  validateXy(X, y);
  const nBins = options.nBins ?? 10;
  if (!Number.isInteger(nBins) || nBins < 2) {
    throw new Error(`nBins must be an integer >= 2. Got ${nBins}.`);
  }
  const yBins = discretize(y, nBins);

  const nFeatures = X[0].length;
  const scores = new Array<number>(nFeatures).fill(0);
  for (let feature = 0; feature < nFeatures; feature += 1) {
    const xColumn = X.map((row) => row[feature]);
    const xBins = discretize(xColumn, nBins);
    scores[feature] = mutualInformationFromDiscretePairs(xBins, yBins);
  }
  return scores;
}

export class SelectFromModel {
  estimator_: ImportanceEstimatorLike | null = null;
  threshold_: number | null = null;
  support_: boolean[] | null = null;
  importances_: Vector | null = null;
  nFeaturesIn_: number | null = null;

  private estimatorFactory: (() => ImportanceEstimatorLike) | ImportanceEstimatorLike;
  private threshold: number | "mean" | "median";
  private maxFeatures?: number;
  private prefit: boolean;
  private importanceGetter?: (estimator: ImportanceEstimatorLike) => Vector;

  constructor(
    estimatorFactory: (() => ImportanceEstimatorLike) | ImportanceEstimatorLike,
    options: SelectFromModelOptions = {},
  ) {
    this.estimatorFactory = estimatorFactory;
    this.threshold = options.threshold ?? "mean";
    this.maxFeatures = options.maxFeatures;
    this.prefit = options.prefit ?? false;
    this.importanceGetter = options.importanceGetter;
  }

  fit(X: Matrix, y: Vector, sampleWeight?: Vector): this {
    validateXy(X, y);
    const estimator = this.prefit
      ? (this.estimatorFactory as ImportanceEstimatorLike)
      : resolveEstimator(this.estimatorFactory);

    if (!this.prefit) {
      estimator.fit(X, y, sampleWeight);
    }
    const importances = extractImportances(estimator, this.importanceGetter);
    if (importances.length !== X[0].length) {
      throw new Error(
        `Estimator importance size mismatch. Expected ${X[0].length}, got ${importances.length}.`,
      );
    }

    const threshold = estimateThreshold(importances, this.threshold);
    let support = importances.map((value) => value >= threshold);
    if (this.maxFeatures !== undefined) {
      if (!Number.isInteger(this.maxFeatures) || this.maxFeatures < 1) {
        throw new Error(`maxFeatures must be an integer >= 1. Got ${this.maxFeatures}.`);
      }
      const top = new Set(rankFeatureIndices(importances, this.maxFeatures, true));
      support = support.map((value, index) => value && top.has(index));
    }
    if (!support.some(Boolean)) {
      const best = rankFeatureIndices(importances, 1, true)[0];
      support[best] = true;
    }

    this.estimator_ = estimator;
    this.importances_ = importances.slice();
    this.threshold_ = threshold;
    this.support_ = support;
    this.nFeaturesIn_ = X[0].length;
    return this;
  }

  transform(X: Matrix): Matrix {
    if (!this.support_ || this.nFeaturesIn_ === null) {
      throw new Error("SelectFromModel has not been fitted.");
    }
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }
    const selected = this.getSupport(true) as number[];
    return subsetColumns(X, selected);
  }

  fitTransform(X: Matrix, y: Vector, sampleWeight?: Vector): Matrix {
    return this.fit(X, y, sampleWeight).transform(X);
  }

  getSupport(indices = false): boolean[] | number[] {
    if (!this.support_) {
      throw new Error("SelectFromModel has not been fitted.");
    }
    if (!indices) {
      return this.support_.slice();
    }
    const selected: number[] = [];
    for (let i = 0; i < this.support_.length; i += 1) {
      if (this.support_[i]) {
        selected.push(i);
      }
    }
    return selected;
  }
}

export class RFE {
  estimator_: ImportanceEstimatorLike | null = null;
  support_: boolean[] | null = null;
  ranking_: number[] | null = null;
  selectedFeatureIndices_: number[] | null = null;
  nFeaturesIn_: number | null = null;

  private estimatorFactory: (() => ImportanceEstimatorLike) | ImportanceEstimatorLike;
  private nFeaturesToSelect?: number;
  private step: number;
  private importanceGetter?: (estimator: ImportanceEstimatorLike) => Vector;

  constructor(
    estimatorFactory: (() => ImportanceEstimatorLike) | ImportanceEstimatorLike,
    options: RFEOptions = {},
  ) {
    this.estimatorFactory = estimatorFactory;
    this.nFeaturesToSelect = options.nFeaturesToSelect;
    this.step = options.step ?? 1;
    this.importanceGetter = options.importanceGetter;
  }

  fit(X: Matrix, y: Vector, sampleWeight?: Vector): this {
    validateXy(X, y);
    const totalFeatures = X[0].length;
    const target = resolveNFeaturesToSelect(this.nFeaturesToSelect, totalFeatures);
    const ranking = new Array<number>(totalFeatures).fill(1);
    let rankValue = 2;

    let current = Array.from({ length: totalFeatures }, (_, idx) => idx);
    while (current.length > target) {
      const estimator = resolveEstimator(this.estimatorFactory);
      const XSubset = subsetColumns(X, current);
      estimator.fit(XSubset, y, sampleWeight);
      const importances = extractImportances(estimator, this.importanceGetter);
      if (importances.length !== current.length) {
        throw new Error(
          `Estimator importance size mismatch. Expected ${current.length}, got ${importances.length}.`,
        );
      }

      const stepCount = Math.min(
        current.length - target,
        resolveStep(this.step, current.length),
      );
      const removeLocalIndices = rankFeatureIndices(importances, stepCount, false);
      const removeSet = new Set(removeLocalIndices);
      const nextCurrent: number[] = [];
      for (let localIndex = 0; localIndex < current.length; localIndex += 1) {
        const globalFeature = current[localIndex];
        if (removeSet.has(localIndex)) {
          ranking[globalFeature] = rankValue;
        } else {
          nextCurrent.push(globalFeature);
        }
      }
      rankValue += 1;
      current = nextCurrent;
    }

    const finalEstimator = resolveEstimator(this.estimatorFactory);
    finalEstimator.fit(subsetColumns(X, current), y, sampleWeight);

    const support = new Array<boolean>(totalFeatures).fill(false);
    for (let i = 0; i < current.length; i += 1) {
      support[current[i]] = true;
    }

    this.estimator_ = finalEstimator;
    this.support_ = support;
    this.ranking_ = ranking;
    this.selectedFeatureIndices_ = current.slice();
    this.nFeaturesIn_ = totalFeatures;
    return this;
  }

  transform(X: Matrix): Matrix {
    if (!this.support_ || this.selectedFeatureIndices_ === null || this.nFeaturesIn_ === null) {
      throw new Error("RFE has not been fitted.");
    }
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }
    return subsetColumns(X, this.selectedFeatureIndices_);
  }

  fitTransform(X: Matrix, y: Vector, sampleWeight?: Vector): Matrix {
    return this.fit(X, y, sampleWeight).transform(X);
  }

  getSupport(indices = false): boolean[] | number[] {
    if (!this.support_ || !this.selectedFeatureIndices_) {
      throw new Error("RFE has not been fitted.");
    }
    if (indices) {
      return this.selectedFeatureIndices_.slice();
    }
    return this.support_.slice();
  }
}

function buildCandidateFeatureCounts(
  nFeatures: number,
  minFeaturesToSelect: number,
  step: number,
): number[] {
  const counts = new Set<number>();
  let current = nFeatures;
  counts.add(current);
  while (current > minFeaturesToSelect) {
    const stepCount = Math.min(current - minFeaturesToSelect, resolveStep(step, current));
    current -= stepCount;
    counts.add(current);
  }
  return Array.from(counts).sort((a, b) => a - b);
}

export class RFECV {
  estimator_: ImportanceEstimatorLike | null = null;
  support_: boolean[] | null = null;
  ranking_: number[] | null = null;
  selectedFeatureIndices_: number[] | null = null;
  nFeatures_: number | null = null;
  cvResults_: Record<string, number[]> = {};
  gridScores_: number[] | null = null;
  nFeaturesIn_: number | null = null;

  private estimatorFactory: (() => ImportanceEstimatorLike) | ImportanceEstimatorLike;
  private cv?: number | CrossValSplitter;
  private scoring?: BuiltInScoring | ScoringFn;
  private minFeaturesToSelect: number;
  private step: number;
  private importanceGetter?: (estimator: ImportanceEstimatorLike) => Vector;

  constructor(
    estimatorFactory: (() => ImportanceEstimatorLike) | ImportanceEstimatorLike,
    options: RFECVOptions = {},
  ) {
    this.estimatorFactory = estimatorFactory;
    this.cv = options.cv;
    this.scoring = options.scoring;
    this.minFeaturesToSelect = options.minFeaturesToSelect ?? 1;
    this.step = options.step ?? 1;
    this.importanceGetter = options.importanceGetter;

    if (!Number.isInteger(this.minFeaturesToSelect) || this.minFeaturesToSelect < 1) {
      throw new Error(
        `minFeaturesToSelect must be an integer >= 1. Got ${this.minFeaturesToSelect}.`,
      );
    }
  }

  fit(X: Matrix, y: Vector, sampleWeight?: Vector): this {
    validateXy(X, y);
    if (sampleWeight) {
      assertVectorLength(sampleWeight, X.length, "sampleWeight");
      assertFiniteVector(sampleWeight, "sampleWeight");
    }

    const nFeatures = X[0].length;
    if (this.minFeaturesToSelect > nFeatures) {
      throw new Error(
        `minFeaturesToSelect (${this.minFeaturesToSelect}) cannot exceed feature count (${nFeatures}).`,
      );
    }

    const candidateCounts = buildCandidateFeatureCounts(
      nFeatures,
      this.minFeaturesToSelect,
      this.step,
    );

    const meanScores: number[] = [];
    let bestScore = Number.NEGATIVE_INFINITY;
    let bestCount = candidateCounts[0];

    for (let i = 0; i < candidateCounts.length; i += 1) {
      const candidate = candidateCounts[i];
      const rfe = new RFE(this.estimatorFactory, {
        nFeaturesToSelect: candidate,
        step: this.step,
        importanceGetter: this.importanceGetter,
      }).fit(X, y, sampleWeight);

      const selected = rfe.getSupport(true) as number[];
      const XSelected = subsetColumns(X, selected);
      const scores = crossValScore(
        () => resolveEstimator(this.estimatorFactory) as unknown as {
          fit(XInner: Matrix, yInner: Vector, sampleWeightInner?: Vector): unknown;
          predict(XInner: Matrix): Vector;
          score?(XInner: Matrix, yInner: Vector): number;
        },
        XSelected,
        y,
        {
          cv: this.cv,
          scoring: this.scoring,
          sampleWeight,
        },
      );
      const score = mean(scores);
      meanScores.push(score);

      if (score > bestScore || (score === bestScore && candidate < bestCount)) {
        bestScore = score;
        bestCount = candidate;
      }
    }

    const finalRFE = new RFE(this.estimatorFactory, {
      nFeaturesToSelect: bestCount,
      step: this.step,
      importanceGetter: this.importanceGetter,
    }).fit(X, y, sampleWeight);

    this.estimator_ = finalRFE.estimator_;
    this.support_ = finalRFE.support_!.slice();
    this.ranking_ = finalRFE.ranking_!.slice();
    this.selectedFeatureIndices_ = (finalRFE.getSupport(true) as number[]).slice();
    this.nFeatures_ = bestCount;
    this.cvResults_ = {
      nFeatures: candidateCounts,
      meanTestScore: meanScores,
    };
    this.gridScores_ = meanScores.slice();
    this.nFeaturesIn_ = nFeatures;
    return this;
  }

  transform(X: Matrix): Matrix {
    if (!this.selectedFeatureIndices_ || this.nFeaturesIn_ === null) {
      throw new Error("RFECV has not been fitted.");
    }
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }
    return subsetColumns(X, this.selectedFeatureIndices_);
  }

  fitTransform(X: Matrix, y: Vector, sampleWeight?: Vector): Matrix {
    return this.fit(X, y, sampleWeight).transform(X);
  }

  getSupport(indices = false): boolean[] | number[] {
    if (!this.support_ || !this.selectedFeatureIndices_) {
      throw new Error("RFECV has not been fitted.");
    }
    if (indices) {
      return this.selectedFeatureIndices_.slice();
    }
    return this.support_.slice();
  }
}
