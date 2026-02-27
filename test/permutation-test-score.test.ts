import { expect, test } from "bun:test";
import { LogisticRegression, permutationTestScore } from "../src";

test("permutationTestScore returns stronger baseline than permuted labels", () => {
  const X = [
    [-3],
    [-2],
    [-1],
    [1],
    [2],
    [3],
    [4],
    [5],
  ];
  const y = [0, 0, 0, 1, 1, 1, 1, 1];

  const result = permutationTestScore(
    () =>
      new LogisticRegression({
        maxIter: 400,
        learningRate: 0.1,
        tolerance: 1e-6,
      }),
    X,
    y,
    {
      cv: 3,
      scoring: "accuracy",
      nPermutations: 20,
      randomState: 42,
    },
  );

  expect(result.permutationScores.length).toBe(20);
  expect(result.score).toBeGreaterThan(0.7);
  expect(result.pValue).toBeLessThan(0.2);
});
