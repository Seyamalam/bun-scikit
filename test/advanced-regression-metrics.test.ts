import { expect, test } from "bun:test";
import {
  explainedVarianceScore,
  meanAbsolutePercentageError,
} from "../src/metrics/regression";

test("advanced regression metrics compute expected values", () => {
  const yTrue = [100, 200, 300, 400];
  const yPred = [90, 210, 290, 420];

  expect(meanAbsolutePercentageError(yTrue, yPred)).toBeCloseTo(0.0583333333, 8);
  expect(explainedVarianceScore(yTrue, yPred)).toBeCloseTo(0.9865, 8);
});
