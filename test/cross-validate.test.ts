import { expect, test } from "bun:test";
import { GroupKFold } from "../src/model_selection/GroupKFold";
import { crossValidate } from "../src/model_selection/crossValidate";
import { SGDClassifier } from "../src/linear_model/SGDClassifier";

test("crossValidate returns fit/score timings and test scores", () => {
  const X = [[-2], [-1], [-0.5], [0.5], [1], [2], [2.5], [3]];
  const y = [0, 0, 0, 1, 1, 1, 1, 1];

  const result = crossValidate(
    () => new SGDClassifier({ loss: "hinge", maxIter: 4000, learningRate: 0.1 }),
    X,
    y,
    { cv: 3, scoring: "accuracy", returnTrainScore: true },
  );

  expect(result.fitTime.length).toBe(3);
  expect(result.scoreTime.length).toBe(3);
  expect(result.testScore.length).toBe(3);
  expect(result.trainScore?.length).toBe(3);
  for (const score of result.testScore) {
    expect(score).toBeGreaterThanOrEqual(0.75);
  }
});

test("crossValidate supports group-aware splitters", () => {
  const X = [
    [0], [0.2], [1], [1.2], [2], [2.2], [3], [3.2], [4], [4.2], [5], [5.2],
  ];
  const y = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1];
  const groups = [10, 10, 11, 11, 20, 20, 21, 21, 30, 30, 31, 31];

  const result = crossValidate(
    () => new SGDClassifier({ loss: "hinge", maxIter: 4000, learningRate: 0.1 }),
    X,
    y,
    { cv: new GroupKFold({ nSplits: 3 }), groups, scoring: "accuracy" },
  );

  expect(result.testScore.length).toBe(3);
});

test("crossValidate accepts sampleWeight and returns scores", () => {
  const X = [[-2], [-1], [-0.5], [0.5], [1], [2], [2.5], [3]];
  const y = [0, 0, 0, 1, 1, 1, 1, 1];
  const sampleWeight = [1, 1, 1, 2, 2, 2, 2, 2];

  const result = crossValidate(
    () => new SGDClassifier({ loss: "hinge", maxIter: 4000, learningRate: 0.1 }),
    X,
    y,
    { cv: 3, scoring: "accuracy", sampleWeight },
  );

  expect(result.testScore.length).toBe(3);
  for (const score of result.testScore) {
    expect(score).toBeGreaterThanOrEqual(0.5);
  }
});
