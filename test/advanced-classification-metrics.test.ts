import { expect, test } from "bun:test";
import {
  balancedAccuracyScore,
  brierScoreLoss,
  matthewsCorrcoef,
} from "../src/metrics/classification";

test("advanced classification metrics return expected values", () => {
  const yTrue = [1, 0, 1, 1, 0, 0, 1, 0];
  const yPred = [1, 0, 1, 0, 0, 1, 1, 0];
  const yProb = [0.9, 0.1, 0.85, 0.45, 0.2, 0.7, 0.8, 0.1];

  expect(balancedAccuracyScore(yTrue, yPred)).toBeCloseTo(0.75, 8);
  expect(matthewsCorrcoef(yTrue, yPred)).toBeCloseTo(0.5, 8);
  expect(brierScoreLoss(yTrue, yProb)).toBeCloseTo(0.115625, 8);
});
