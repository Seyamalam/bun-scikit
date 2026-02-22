import { expect, test } from "bun:test";
import {
  LinearRegression,
  LogisticRegression,
  Pipeline,
  StandardScaler,
  crossValScore,
} from "../src";

test("crossValScore works for classification with explicit accuracy scoring", () => {
  const X = [
    [-3], [-2], [-1], [-0.5], [0.5], [1], [2], [3],
    [-2.5], [-1.5], [1.5], [2.5],
  ];
  const y = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1];

  const scores = crossValScore(
    () =>
      new Pipeline([
        ["scaler", new StandardScaler()],
        [
          "classifier",
          new LogisticRegression({
            solver: "gd",
            learningRate: 0.8,
            maxIter: 20,
            tolerance: 1e-4,
            l2: 0.01,
          }),
        ],
      ]),
    X,
    y,
    { cv: 4, scoring: "accuracy" },
  );

  expect(scores.length).toBe(4);
  for (const score of scores) {
    expect(score).toBeGreaterThanOrEqual(0.9);
  }
});

test("crossValScore works for regression with r2 scoring", () => {
  const X = [[0], [1], [2], [3], [4], [5], [6], [7]];
  const y = [1, 3, 5, 7, 9, 11, 13, 15];

  const scores = crossValScore(
    () => new LinearRegression({ solver: "normal" }),
    X,
    y,
    { cv: 4, scoring: "r2" },
  );

  expect(scores.length).toBe(4);
  for (const score of scores) {
    expect(score).toBeGreaterThan(0.999999);
  }
});

test("crossValScore defaults to estimator.score when scoring is omitted", () => {
  class ScoreOnlyEstimator {
    fit(_X: number[][], _y: number[]): this {
      return this;
    }
    predict(_X: number[][]): never {
      throw new Error("predict should not be called when score() is available and scoring is omitted.");
    }
    score(_X: number[][], _y: number[]): number {
      return 0.1234;
    }
  }

  const X = [[1], [2], [3], [4], [5], [6]];
  const y = [10, 20, 30, 40, 50, 60];

  const scores = crossValScore(() => new ScoreOnlyEstimator(), X, y, { cv: 3 });
  expect(scores).toEqual([0.1234, 0.1234, 0.1234]);
});
