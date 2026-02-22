import { expect, test } from "bun:test";
import { StratifiedShuffleSplit } from "../src/model_selection/StratifiedShuffleSplit";

test("StratifiedShuffleSplit preserves split sizes and class ratio approximately", () => {
  const X = Array.from({ length: 20 }, (_, i) => i);
  const y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];

  const splitter = new StratifiedShuffleSplit({
    nSplits: 3,
    testSize: 0.25,
    randomState: 7,
  });
  const splits = splitter.split(X, y);

  expect(splits.length).toBe(3);
  for (const split of splits) {
    expect(split.testIndices.length).toBe(5);
    expect(split.trainIndices.length).toBe(15);

    const overlap = split.trainIndices.filter((idx) => split.testIndices.includes(idx));
    expect(overlap.length).toBe(0);

    const positives = split.testIndices.map((idx) => y[idx]).filter((v) => v === 1).length;
    expect(positives === 1 || positives === 2).toBeTrue();
  }
});

test("StratifiedShuffleSplit is deterministic for fixed randomState", () => {
  const X = Array.from({ length: 20 }, (_, i) => i);
  const y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];

  const a = new StratifiedShuffleSplit({
    nSplits: 4,
    testSize: 0.2,
    randomState: 17,
  }).split(X, y);
  const b = new StratifiedShuffleSplit({
    nSplits: 4,
    testSize: 0.2,
    randomState: 17,
  }).split(X, y);

  expect(a).toEqual(b);
});

test("StratifiedShuffleSplit rejects classes with fewer than two members", () => {
  const X = Array.from({ length: 6 }, (_, i) => i);
  const y = [0, 0, 0, 0, 0, 1];

  expect(() => new StratifiedShuffleSplit({ testSize: 0.5 }).split(X, y)).toThrow(
    /least populated class/i,
  );
});
