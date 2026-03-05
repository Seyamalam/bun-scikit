import { expect, test } from "bun:test";
import {
  LeaveOneGroupOut,
  LeaveOneOut,
  LeavePGroupsOut,
  LeavePOut,
  PredefinedSplit,
  ShuffleSplit,
  TimeSeriesSplit,
} from "../src";

test("ShuffleSplit produces deterministic train/test partitions", () => {
  const X = Array.from({ length: 10 }, (_, index) => index);
  const a = new ShuffleSplit({ nSplits: 3, testSize: 0.2, randomState: 9 }).split(X);
  const b = new ShuffleSplit({ nSplits: 3, testSize: 0.2, randomState: 9 }).split(X);

  expect(a).toEqual(b);
  expect(a).toHaveLength(3);
  for (const split of a) {
    expect(split.testIndices).toHaveLength(2);
    expect(split.trainIndices).toHaveLength(8);
    expect(split.trainIndices.filter((index) => split.testIndices.includes(index))).toHaveLength(0);
  }
});

test("LeaveOneOut leaves each sample out exactly once", () => {
  const X = Array.from({ length: 4 }, (_, index) => index);
  const splits = new LeaveOneOut().split(X);

  expect(splits).toHaveLength(4);
  expect(splits.map((split) => split.testIndices[0])).toEqual([0, 1, 2, 3]);
  for (const split of splits) {
    expect(split.testIndices).toHaveLength(1);
    expect(split.trainIndices).toHaveLength(3);
  }
});

test("LeavePOut enumerates p-sized holdout combinations", () => {
  const X = Array.from({ length: 4 }, (_, index) => index);
  const splitter = new LeavePOut({ p: 2 });
  const splits = splitter.split(X);

  expect(splitter.getNSplits(X)).toBe(6);
  expect(splits).toHaveLength(6);
  expect(splits[0]).toEqual({ trainIndices: [2, 3], testIndices: [0, 1] });
  expect(splits.at(-1)).toEqual({ trainIndices: [0, 1], testIndices: [2, 3] });
});

test("LeaveOneGroupOut keeps group members together in the held-out fold", () => {
  const X = Array.from({ length: 6 }, (_, index) => index);
  const groups = [0, 0, 1, 1, 2, 2];
  const splitter = new LeaveOneGroupOut();
  const splits = splitter.split(X, undefined, groups);

  expect(splitter.getNSplits(X, undefined, groups)).toBe(3);
  expect(splits).toEqual([
    { trainIndices: [2, 3, 4, 5], testIndices: [0, 1] },
    { trainIndices: [0, 1, 4, 5], testIndices: [2, 3] },
    { trainIndices: [0, 1, 2, 3], testIndices: [4, 5] },
  ]);
});

test("LeavePGroupsOut enumerates combinations of held-out groups", () => {
  const X = Array.from({ length: 6 }, (_, index) => index);
  const groups = [0, 0, 1, 1, 2, 2];
  const splitter = new LeavePGroupsOut({ p: 2 });
  const splits = splitter.split(X, undefined, groups);

  expect(splitter.getNSplits(X, undefined, groups)).toBe(3);
  expect(splits).toEqual([
    { trainIndices: [4, 5], testIndices: [0, 1, 2, 3] },
    { trainIndices: [2, 3], testIndices: [0, 1, 4, 5] },
    { trainIndices: [0, 1], testIndices: [2, 3, 4, 5] },
  ]);
});

test("PredefinedSplit respects explicit fold assignments and -1 train-only rows", () => {
  const X = Array.from({ length: 6 }, (_, index) => index);
  const splitter = new PredefinedSplit({ testFold: [0, 0, 1, 1, -1, -1] });
  const splits = splitter.split(X);

  expect(splitter.getNSplits()).toBe(2);
  expect(splits).toEqual([
    { trainIndices: [2, 3, 4, 5], testIndices: [0, 1] },
    { trainIndices: [0, 1, 4, 5], testIndices: [2, 3] },
  ]);
});

test("TimeSeriesSplit grows the training window by default", () => {
  const X = Array.from({ length: 12 }, (_, index) => index);
  const splitter = new TimeSeriesSplit({ nSplits: 3 });
  const splits = splitter.split(X);

  expect(splitter.getNSplits()).toBe(3);
  expect(splits).toEqual([
    { trainIndices: [0, 1, 2], testIndices: [3, 4, 5] },
    { trainIndices: [0, 1, 2, 3, 4, 5], testIndices: [6, 7, 8] },
    { trainIndices: [0, 1, 2, 3, 4, 5, 6, 7, 8], testIndices: [9, 10, 11] },
  ]);
});

test("TimeSeriesSplit supports gap and maxTrainSize", () => {
  const X = Array.from({ length: 10 }, (_, index) => index);
  const splits = new TimeSeriesSplit({
    nSplits: 2,
    testSize: 2,
    gap: 1,
    maxTrainSize: 3,
  }).split(X);

  expect(splits).toEqual([
    { trainIndices: [2, 3, 4], testIndices: [6, 7] },
    { trainIndices: [4, 5, 6], testIndices: [8, 9] },
  ]);
});
