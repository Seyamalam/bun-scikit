import { expect, test } from "bun:test";
import { StratifiedGroupKFold } from "../src";

function classRate(values: number[], labels: number[]): number {
  let positives = 0;
  for (let i = 0; i < values.length; i += 1) {
    if (labels[values[i]] === 1) {
      positives += 1;
    }
  }
  return positives / Math.max(1, values.length);
}

test("StratifiedGroupKFold keeps groups together and roughly preserves class ratio", () => {
  const X = Array.from({ length: 24 }, (_, i) => [i]);
  const groups = Array.from({ length: 24 }, (_, i) => Math.floor(i / 2));
  const y = [
    0, 0, 0, 0, 0, 1, 0, 1,
    0, 1, 1, 1, 0, 0, 1, 1,
    0, 1, 0, 1, 1, 1, 0, 1,
  ];

  const splits = new StratifiedGroupKFold({
    nSplits: 4,
    shuffle: true,
    randomState: 13,
  }).split(X, y, groups);

  expect(splits.length).toBe(4);
  const overallRate = y.filter((v) => v === 1).length / y.length;
  for (const split of splits) {
    const inTest = new Set(split.testIndices);
    for (let i = 0; i < groups.length; i += 1) {
      for (let j = i + 1; j < groups.length; j += 1) {
        if (groups[i] === groups[j]) {
          expect(inTest.has(i)).toBe(inTest.has(j));
        }
      }
    }
    const foldRate = classRate(split.testIndices, y);
    expect(Math.abs(foldRate - overallRate)).toBeLessThan(0.25);
  }
});

test("StratifiedGroupKFold is deterministic with fixed randomState", () => {
  const X = Array.from({ length: 18 }, (_, i) => [i]);
  const groups = Array.from({ length: 18 }, (_, i) => Math.floor(i / 3));
  const y = [0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1];

  const a = new StratifiedGroupKFold({
    nSplits: 3,
    shuffle: true,
    randomState: 5,
  }).split(X, y, groups);
  const b = new StratifiedGroupKFold({
    nSplits: 3,
    shuffle: true,
    randomState: 5,
  }).split(X, y, groups);
  expect(a).toEqual(b);
});
