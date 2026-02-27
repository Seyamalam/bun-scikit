import { expect, test } from "bun:test";
import {
  brierScoreLoss,
  confusionMatrix,
  logLoss,
  matthewsCorrcoef,
} from "../src";

test("multiclass confusion and mcc compute finite values", () => {
  const yTrue = [0, 1, 2, 0, 1, 2, 0, 1, 2];
  const yPred = [0, 2, 2, 0, 1, 1, 0, 1, 2];
  const cm = confusionMatrix(yTrue, yPred);
  expect(cm.labels).toEqual([0, 1, 2]);
  expect(cm.matrix).toEqual([
    [3, 0, 0],
    [0, 2, 1],
    [0, 1, 2],
  ]);
  const mcc = matthewsCorrcoef(yTrue, yPred);
  expect(Number.isFinite(mcc)).toBeTrue();
  expect(mcc).toBeGreaterThan(0.4);
});

test("multiclass logLoss and brierScoreLoss accept probability matrices", () => {
  const yTrue = [0, 1, 2, 0, 1, 2];
  const proba = [
    [0.8, 0.1, 0.1],
    [0.1, 0.7, 0.2],
    [0.1, 0.3, 0.6],
    [0.75, 0.2, 0.05],
    [0.2, 0.7, 0.1],
    [0.05, 0.2, 0.75],
  ];

  expect(logLoss(yTrue, proba)).toBeLessThan(0.5);
  expect(brierScoreLoss(yTrue, proba)).toBeLessThan(0.2);
});
