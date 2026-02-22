import { expect, test } from "bun:test";
import {
  meanAbsoluteError,
  meanSquaredError,
  r2Score,
} from "../src/metrics/regression";

test("regression metrics return expected values", () => {
  const yTrue = [3, -0.5, 2, 7];
  const yPred = [2.5, 0, 2, 8];

  expect(meanSquaredError(yTrue, yPred)).toBeCloseTo(0.375, 8);
  expect(meanAbsoluteError(yTrue, yPred)).toBeCloseTo(0.5, 8);
  expect(r2Score(yTrue, yPred)).toBeCloseTo(0.948608, 5);
});
