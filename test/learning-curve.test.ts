import { expect, test } from "bun:test";
import { LogisticRegression, learningCurve } from "../src";

test("learningCurve returns train and test scores per train size", () => {
  const X = [[-2], [-1], [-0.5], [0.5], [1], [2], [2.5], [3], [3.5], [4]];
  const y = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1];

  const result = learningCurve(
    () => new LogisticRegression({ maxIter: 300, learningRate: 0.15, tolerance: 1e-6 }),
    X,
    y,
    { cv: 3, scoring: "accuracy", trainSizes: [0.4, 0.7, 1.0] },
  );

  expect(result.trainSizes.length).toBe(3);
  expect(result.trainScores.length).toBe(3);
  expect(result.testScores.length).toBe(3);
  for (let i = 0; i < result.trainScores.length; i += 1) {
    expect(result.trainScores[i].length).toBe(3);
    expect(result.testScores[i].length).toBe(3);
  }
});
