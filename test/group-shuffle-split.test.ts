import { expect, test } from "bun:test";
import { GroupShuffleSplit } from "../src";

test("GroupShuffleSplit keeps group members together", () => {
  const X = Array.from({ length: 12 }, (_, i) => i);
  const groups = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5];
  const splits = new GroupShuffleSplit({
    nSplits: 3,
    testSize: 0.33,
    randomState: 11,
  }).split(X, undefined, groups);

  expect(splits.length).toBe(3);
  for (const split of splits) {
    const inTest = new Set(split.testIndices);
    for (let i = 0; i < groups.length; i += 1) {
      for (let j = i + 1; j < groups.length; j += 1) {
        if (groups[i] === groups[j]) {
          expect(inTest.has(i)).toBe(inTest.has(j));
        }
      }
    }
    expect(split.trainIndices.length).toBeGreaterThan(0);
    expect(split.testIndices.length).toBeGreaterThan(0);
  }
});

test("GroupShuffleSplit is deterministic with fixed randomState", () => {
  const X = Array.from({ length: 20 }, (_, i) => i);
  const groups = Array.from({ length: 20 }, (_, i) => Math.floor(i / 2));
  const a = new GroupShuffleSplit({
    nSplits: 4,
    testSize: 0.2,
    randomState: 17,
  }).split(X, undefined, groups);
  const b = new GroupShuffleSplit({
    nSplits: 4,
    testSize: 0.2,
    randomState: 17,
  }).split(X, undefined, groups);
  expect(a).toEqual(b);
});
