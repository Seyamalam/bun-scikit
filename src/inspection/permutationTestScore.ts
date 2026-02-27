import type { Matrix, Vector } from "../types";
import { crossValScore, type BuiltInScoring, type CrossValSplitter, type ScoringFn } from "../model_selection/crossValScore";
import { assertFiniteVector, assertVectorLength } from "../utils/validation";

type CrossValEstimator = {
  fit(X: Matrix, y: Vector, sampleWeight?: Vector): unknown;
  predict(X: Matrix): Vector;
  score?(X: Matrix, y: Vector): number;
};

export interface PermutationTestScoreOptions {
  cv?: number | CrossValSplitter;
  scoring?: BuiltInScoring | ScoringFn;
  nPermutations?: number;
  randomState?: number;
  sampleWeight?: Vector;
}

export interface PermutationTestScoreResult {
  score: number;
  permutationScores: Vector;
  pValue: number;
}

function mean(values: number[]): number {
  let sum = 0;
  for (let i = 0; i < values.length; i += 1) {
    sum += values[i];
  }
  return values.length === 0 ? 0 : sum / values.length;
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

function shuffleCopy(values: Vector, seed: number): Vector {
  const out = values.slice();
  const random = mulberry32(seed);
  for (let i = out.length - 1; i > 0; i -= 1) {
    const j = Math.floor(random() * (i + 1));
    const tmp = out[i];
    out[i] = out[j];
    out[j] = tmp;
  }
  return out;
}

export function permutationTestScore(
  createEstimator: () => CrossValEstimator,
  X: Matrix,
  y: Vector,
  options: PermutationTestScoreOptions = {},
): PermutationTestScoreResult {
  if (typeof createEstimator !== "function") {
    throw new Error("createEstimator must be a function returning a new estimator.");
  }
  if (!Array.isArray(X) || X.length === 0) {
    throw new Error("X must be a non-empty matrix.");
  }
  assertVectorLength(y, X.length);
  assertFiniteVector(y);
  if (options.sampleWeight) {
    assertVectorLength(options.sampleWeight, X.length, "sampleWeight");
    assertFiniteVector(options.sampleWeight, "sampleWeight");
  }
  const nPermutations = options.nPermutations ?? 100;
  if (!Number.isInteger(nPermutations) || nPermutations < 1) {
    throw new Error(`nPermutations must be an integer >= 1. Got ${nPermutations}.`);
  }
  const randomState = options.randomState ?? 0;
  if (!Number.isInteger(randomState)) {
    throw new Error(`randomState must be an integer. Got ${randomState}.`);
  }

  const baselineScores = crossValScore(createEstimator, X, y, {
    cv: options.cv,
    scoring: options.scoring,
    sampleWeight: options.sampleWeight,
  });
  const baseline = mean(baselineScores);

  const permutationScores = new Array<number>(nPermutations);
  let geCount = 1; // +1 smoothing
  for (let i = 0; i < nPermutations; i += 1) {
    const yPerm = shuffleCopy(y, randomState + i * 104_729);
    const scores = crossValScore(createEstimator, X, yPerm, {
      cv: options.cv,
      scoring: options.scoring,
      sampleWeight: options.sampleWeight,
    });
    const current = mean(scores);
    permutationScores[i] = current;
    if (current >= baseline) {
      geCount += 1;
    }
  }

  const pValue = geCount / (nPermutations + 1);
  return {
    score: baseline,
    permutationScores,
    pValue,
  };
}
