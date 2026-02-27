import { expect, test } from "bun:test";
import { LogisticRegression, validationCurve } from "../src";

test("validationCurve evaluates scores across parameter values", () => {
  const X = [[-2], [-1], [-0.5], [0.5], [1], [2], [2.5], [3], [3.5], [4]];
  const y = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1];

  const result = validationCurve(
    () => new LogisticRegression({ maxIter: 100, learningRate: 0.1, tolerance: 1e-5 }),
    X,
    y,
    {
      cv: 3,
      scoring: "accuracy",
      paramName: "maxIter",
      paramRange: [20, 60, 100],
    },
  );

  expect(result.paramRange).toEqual([20, 60, 100]);
  expect(result.trainScores.length).toBe(3);
  expect(result.testScores.length).toBe(3);
  for (let i = 0; i < result.trainScores.length; i += 1) {
    expect(result.trainScores[i].length).toBe(3);
    expect(result.testScores[i].length).toBe(3);
  }
});
