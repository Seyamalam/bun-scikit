import type { Matrix, Vector } from "../types";
import { accuracyScore, f1Score, precisionScore, recallScore } from "../metrics/classification";
import { meanSquaredError, r2Score } from "../metrics/regression";
import {
  crossValScore,
  type BuiltInScoring,
  type CrossValEstimator,
  type CrossValSplitter,
  type ScoringFn,
} from "./crossValScore";

export type ParamDistributions = Record<string, readonly unknown[]>;

export interface RandomizedSearchCVOptions {
  cv?: number | CrossValSplitter;
  scoring?: BuiltInScoring | ScoringFn;
  refit?: boolean;
  errorScore?: "raise" | number;
  nIter?: number;
  randomState?: number;
}

export interface RandomizedSearchResultRow {
  params: Record<string, unknown>;
  splitScores: number[];
  meanTestScore: number;
  stdTestScore: number;
  rank: number;
  status: "ok" | "error";
  errorMessage?: string;
}

function mean(values: number[]): number {
  let sum = 0;
  for (let i = 0; i < values.length; i += 1) {
    sum += values[i];
  }
  return sum / values.length;
}

function std(values: number[]): number {
  if (values.length < 2) {
    return 0;
  }
  const avg = mean(values);
  let sum = 0;
  for (let i = 0; i < values.length; i += 1) {
    const diff = values[i] - avg;
    sum += diff * diff;
  }
  return Math.sqrt(sum / values.length);
}

function resolveBuiltInScorer(scoring: BuiltInScoring): ScoringFn {
  switch (scoring) {
    case "accuracy":
      return accuracyScore;
    case "f1":
      return f1Score;
    case "precision":
      return precisionScore;
    case "recall":
      return recallScore;
    case "r2":
      return r2Score;
    case "mean_squared_error":
      return meanSquaredError;
    case "neg_mean_squared_error":
      return (yTrue, yPred) => -meanSquaredError(yTrue, yPred);
    default: {
      const exhaustive: never = scoring;
      throw new Error(`Unsupported scoring metric: ${exhaustive}`);
    }
  }
}

function isLossMetric(scoring: BuiltInScoring | ScoringFn | undefined): boolean {
  return scoring === "mean_squared_error";
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

  nextInt(maxExclusive: number): number {
    return Math.floor(this.next() * maxExclusive);
  }
}

function sampleParams(
  distributions: ParamDistributions,
  nIter: number,
  randomState: number,
): Record<string, unknown>[] {
  const keys = Object.keys(distributions);
  if (keys.length === 0) {
    throw new Error("paramDistributions must include at least one parameter.");
  }
  for (let i = 0; i < keys.length; i += 1) {
    const values = distributions[keys[i]];
    if (!Array.isArray(values) || values.length === 0) {
      throw new Error(`paramDistributions '${keys[i]}' must be a non-empty array.`);
    }
  }

  const rng = new Mulberry32(randomState);
  const out: Record<string, unknown>[] = [];
  for (let i = 0; i < nIter; i += 1) {
    const params: Record<string, unknown> = {};
    for (let k = 0; k < keys.length; k += 1) {
      const key = keys[k];
      const values = distributions[key];
      params[key] = values[rng.nextInt(values.length)];
    }
    out.push(params);
  }
  return out;
}

export class RandomizedSearchCV<TEstimator extends CrossValEstimator> {
  bestEstimator_: TEstimator | null = null;
  bestParams_: Record<string, unknown> | null = null;
  bestScore_: number | null = null;
  cvResults_: RandomizedSearchResultRow[] = [];

  private readonly estimatorFactory: (params: Record<string, unknown>) => TEstimator;
  private readonly paramDistributions: ParamDistributions;
  private readonly cv?: number | CrossValSplitter;
  private readonly scoring?: BuiltInScoring | ScoringFn;
  private readonly refit: boolean;
  private readonly errorScore: "raise" | number;
  private readonly nIter: number;
  private readonly randomState: number;
  private isFitted = false;

  constructor(
    estimatorFactory: (params: Record<string, unknown>) => TEstimator,
    paramDistributions: ParamDistributions,
    options: RandomizedSearchCVOptions = {},
  ) {
    if (typeof estimatorFactory !== "function") {
      throw new Error("estimatorFactory must be a function.");
    }
    this.estimatorFactory = estimatorFactory;
    this.paramDistributions = paramDistributions;
    this.cv = options.cv;
    this.scoring = options.scoring;
    this.refit = options.refit ?? true;
    this.errorScore = options.errorScore ?? "raise";
    this.nIter = options.nIter ?? 10;
    this.randomState = options.randomState ?? 42;
    if (!Number.isInteger(this.nIter) || this.nIter < 1) {
      throw new Error(`nIter must be an integer >= 1. Got ${this.nIter}.`);
    }
  }

  fit(X: Matrix, y: Vector): this {
    const candidates = sampleParams(this.paramDistributions, this.nIter, this.randomState);
    const minimize = isLossMetric(this.scoring);
    const rows: RandomizedSearchResultRow[] = [];
    const objectiveScores: number[] = [];

    for (let candidateIndex = 0; candidateIndex < candidates.length; candidateIndex += 1) {
      const params = candidates[candidateIndex];
      try {
        const splitScores = crossValScore(
          () => this.estimatorFactory(params),
          X,
          y,
          { cv: this.cv, scoring: this.scoring },
        );
        const meanTestScore = mean(splitScores);
        rows.push({
          params: { ...params },
          splitScores,
          meanTestScore,
          stdTestScore: std(splitScores),
          rank: 0,
          status: "ok",
        });
        objectiveScores.push(minimize ? -meanTestScore : meanTestScore);
      } catch (error) {
        if (this.errorScore === "raise") {
          throw error;
        }
        rows.push({
          params: { ...params },
          splitScores: [this.errorScore],
          meanTestScore: this.errorScore,
          stdTestScore: 0,
          rank: 0,
          status: "error",
          errorMessage: error instanceof Error ? error.message : String(error),
        });
        objectiveScores.push(minimize ? -this.errorScore : this.errorScore);
      }
    }

    const order = Array.from({ length: rows.length }, (_, idx) => idx).sort((a, b) => {
      const delta = objectiveScores[b] - objectiveScores[a];
      if (delta !== 0) {
        return delta;
      }
      return a - b;
    });

    for (let rank = 0; rank < order.length; rank += 1) {
      rows[order[rank]].rank = rank + 1;
    }

    const bestIndex = order[0];
    this.bestParams_ = { ...rows[bestIndex].params };
    this.bestScore_ = rows[bestIndex].meanTestScore;
    this.cvResults_ = rows;

    if (this.refit) {
      const estimator = this.estimatorFactory(this.bestParams_);
      estimator.fit(X, y);
      this.bestEstimator_ = estimator;
    } else {
      this.bestEstimator_ = null;
    }

    this.isFitted = true;
    return this;
  }

  predict(X: Matrix): Vector {
    if (!this.isFitted) {
      throw new Error("RandomizedSearchCV has not been fitted.");
    }
    if (!this.refit || !this.bestEstimator_) {
      throw new Error("RandomizedSearchCV predict is unavailable when refit=false.");
    }
    return this.bestEstimator_.predict(X);
  }

  score(X: Matrix, y: Vector): number {
    if (!this.isFitted) {
      throw new Error("RandomizedSearchCV has not been fitted.");
    }
    if (!this.refit || !this.bestEstimator_) {
      throw new Error("RandomizedSearchCV score is unavailable when refit=false.");
    }

    if (this.scoring) {
      const scorer =
        typeof this.scoring === "function" ? this.scoring : resolveBuiltInScorer(this.scoring);
      return scorer(y, this.bestEstimator_.predict(X));
    }

    if (typeof this.bestEstimator_.score === "function") {
      return this.bestEstimator_.score(X, y);
    }

    throw new Error("No scoring function available. Provide scoring in RandomizedSearchCV options.");
  }
}
