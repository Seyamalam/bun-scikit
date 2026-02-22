import { expect, test } from "bun:test";
import {
  accuracyScore,
  classificationReport,
  confusionMatrix,
  f1Score,
  logLoss,
  precisionScore,
  recallScore,
  rocAucScore,
} from "../src/metrics/classification";

test("classification metrics compute expected values", () => {
  const yTrue = [1, 0, 1, 1, 0, 1, 0, 0];
  const yPred = [1, 0, 1, 0, 0, 1, 1, 0];

  expect(accuracyScore(yTrue, yPred)).toBeCloseTo(0.75, 8);
  expect(precisionScore(yTrue, yPred)).toBeCloseTo(0.75, 8);
  expect(recallScore(yTrue, yPred)).toBeCloseTo(0.75, 8);
  expect(f1Score(yTrue, yPred)).toBeCloseTo(0.75, 8);
});

test("advanced classification metrics compute expected values", () => {
  const yTrue = [1, 0, 1, 1, 0, 1, 0, 0];
  const yPred = [1, 0, 1, 0, 0, 1, 1, 0];
  const yProb = [0.9, 0.1, 0.8, 0.4, 0.2, 0.75, 0.6, 0.1];

  const cm = confusionMatrix(yTrue, yPred);
  expect(cm.labels).toEqual([0, 1]);
  expect(cm.matrix).toEqual([
    [3, 1],
    [1, 3],
  ]);

  const auc = rocAucScore(yTrue, yProb);
  expect(auc).toBeCloseTo(0.9375, 8);

  const ll = logLoss(yTrue, yProb);
  expect(ll).toBeCloseTo(0.3603290232, 8);

  const report = classificationReport(yTrue, yPred);
  expect(report.accuracy).toBeCloseTo(0.75, 8);
  expect(report.perLabel["0"].precision).toBeCloseTo(0.75, 8);
  expect(report.perLabel["1"].recall).toBeCloseTo(0.75, 8);
  expect(report.macroAvg.f1Score).toBeCloseTo(0.75, 8);
  expect(report.weightedAvg.f1Score).toBeCloseTo(0.75, 8);
});
