import { expect, test } from "bun:test";
import {
  adjustedRandScore,
  calinskiHarabaszScore,
  daviesBouldinScore,
  silhouetteScore,
} from "../src";

test("clustering metrics return sensible values", () => {
  const X = [
    [0, 0],
    [0.1, 0.0],
    [-0.1, 0.1],
    [5, 5],
    [5.1, 5.0],
    [4.9, 5.2],
  ];
  const labels = [0, 0, 0, 1, 1, 1];

  const sil = silhouetteScore(X, labels);
  expect(sil).toBeGreaterThan(0.5);

  const ch = calinskiHarabaszScore(X, labels);
  expect(ch).toBeGreaterThan(1);

  const db = daviesBouldinScore(X, labels);
  expect(db).toBeGreaterThan(0);
  expect(db).toBeLessThan(1);
});

test("adjustedRandScore matches perfect and mismatched partitions", () => {
  const truth = [0, 0, 1, 1, 2, 2];
  const predPerfect = [1, 1, 0, 0, 2, 2];
  const predRandom = [0, 1, 0, 1, 0, 1];
  expect(adjustedRandScore(truth, predPerfect)).toBeCloseTo(1, 10);
  expect(adjustedRandScore(truth, predRandom)).toBeLessThan(0.5);
});
