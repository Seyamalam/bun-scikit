import { expect, test } from "bun:test";
import { accuracyScore } from "../src";

test("accuracyScore supports multilabel indicator matrices", () => {
  const yTrue = [
    [1, 0, 1],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
  ];
  const yPred = [
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
  ];
  const score = accuracyScore(yTrue, yPred);
  expect(score).toBeCloseTo(0.75, 8);
});
