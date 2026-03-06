import { expect, test } from "bun:test";
import { RANSACRegressor } from "../src";

test("RANSACRegressor fits the inlier trend and rejects a large outlier", () => {
  const X = [[0], [1], [2], [3], [4], [5], [6]];
  const y = [1, 3, 5, 7, 100, 11, 13];

  const reg = new RANSACRegressor({
    minSamples: 2,
    residualThreshold: 3,
    maxTrials: 200,
    randomState: 7,
  }).fit(X, y);

  expect(reg.inlierMask_).not.toBeNull();
  expect(reg.inlierMask_![4]).toBeFalse();
  const pred = reg.predict([[7]]);
  expect(pred[0]).toBeGreaterThan(13);
  expect(pred[0]).toBeLessThan(17);
});
