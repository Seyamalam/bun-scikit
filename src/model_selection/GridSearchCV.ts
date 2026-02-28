import type { Matrix, Vector } from "../types";
import {
  crossValScore,
  type BuiltInScoring,
  type CrossValEstimator,
  type CrossValSplitter,
  type ScoringFn,
} from "./crossValScore";
import { resolveBuiltInScorer } from "./shared";
import { expandParamGrid, type ParamGrid, type ParamGridInput } from "./ParameterGrid";

export interface GridSearchCVOptions {
  cv?: number | CrossValSplitter;
  scoring?: BuiltInScoring | ScoringFn;
  refit?: boolean;
  errorScore?: "raise" | number;
}

export interface GridSearchResultRow {
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

function isLossMetric(scoring: BuiltInScoring | ScoringFn | undefined): boolean {
  return scoring === "mean_squared_error";
}

export class GridSearchCV<TEstimator extends CrossValEstimator> {
  bestEstimator_: TEstimator | null = null;
  bestParams_: Record<string, unknown> | null = null;
  bestScore_: number | null = null;
  cvResults_: GridSearchResultRow[] = [];

  private readonly estimatorFactory: (params: Record<string, unknown>) => TEstimator;
  private readonly paramGrid: ParamGridInput;
  private readonly cv?: number | CrossValSplitter;
  private readonly scoring?: BuiltInScoring | ScoringFn;
  private readonly refit: boolean;
  private readonly errorScore: "raise" | number;
  private isFitted = false;

  constructor(
    estimatorFactory: (params: Record<string, unknown>) => TEstimator,
    paramGrid: ParamGridInput,
    options: GridSearchCVOptions = {},
  ) {
    if (typeof estimatorFactory !== "function") {
      throw new Error("estimatorFactory must be a function.");
    }
    this.estimatorFactory = estimatorFactory;
    this.paramGrid = paramGrid;
    this.cv = options.cv;
    this.scoring = options.scoring;
    this.refit = options.refit ?? true;
    this.errorScore = options.errorScore ?? "raise";
  }

  fit(X: Matrix, y: Vector, sampleWeight?: Vector): this {
    const candidates = expandParamGrid(this.paramGrid);
    const minimize = isLossMetric(this.scoring);
    const rows: GridSearchResultRow[] = [];
    const objectiveScores: number[] = [];

    for (let candidateIndex = 0; candidateIndex < candidates.length; candidateIndex += 1) {
      const params = candidates[candidateIndex];
      try {
        const splitScores = crossValScore(
          () => this.estimatorFactory(params),
          X,
          y,
          { cv: this.cv, scoring: this.scoring, sampleWeight },
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
      estimator.fit(X, y, sampleWeight);
      this.bestEstimator_ = estimator;
    } else {
      this.bestEstimator_ = null;
    }

    this.isFitted = true;
    return this;
  }

  predict(X: Matrix): Vector {
    if (!this.isFitted) {
      throw new Error("GridSearchCV has not been fitted.");
    }
    if (!this.refit || !this.bestEstimator_) {
      throw new Error("GridSearchCV predict is unavailable when refit=false.");
    }
    return this.bestEstimator_.predict(X);
  }

  score(X: Matrix, y: Vector): number {
    if (!this.isFitted) {
      throw new Error("GridSearchCV has not been fitted.");
    }
    if (!this.refit || !this.bestEstimator_) {
      throw new Error("GridSearchCV score is unavailable when refit=false.");
    }

    if (this.scoring) {
      const scorer =
        typeof this.scoring === "function" ? this.scoring : resolveBuiltInScorer(this.scoring);
      return scorer(y, this.bestEstimator_.predict(X));
    }

    if (typeof this.bestEstimator_.score === "function") {
      return this.bestEstimator_.score(X, y);
    }

    throw new Error("No scoring function available. Provide scoring in GridSearchCV options.");
  }
}

