import { expect, test } from "bun:test";
import {
  accuracyScore,
  confusionMatrix,
  f1Score,
  precisionScore,
  recallScore,
} from "../src/metrics/classification";

test("classification metrics compute expected values", () => {
  const yTrue = [1, 0, 1, 1, 0, 1, 0, 0];
  const yPred = [1, 0, 1, 0, 0, 1, 1, 0];

  expect(accuracyScore(yTrue, yPred)).toBeCloseTo(0.75, 8);
  expect(precisionScore(yTrue, yPred)).toBeCloseTo(0.75, 8);
  expect(recallScore(yTrue, yPred)).toBeCloseTo(0.75, 8);
  expect(f1Score(yTrue, yPred)).toBeCloseTo(0.75, 8);
  expect(confusionMatrix(yTrue, yPred)).toEqual([
    [3, 1],
    [1, 3],
  ]);
});
