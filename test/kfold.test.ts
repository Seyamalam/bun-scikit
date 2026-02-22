import { expect, test } from "bun:test";
import { KFold } from "../src/model_selection/KFold";
import { StratifiedKFold } from "../src/model_selection/StratifiedKFold";

test("KFold split covers each sample exactly once in test sets", () => {
  const X = Array.from({ length: 10 }, (_, i) => i);
  const splitter = new KFold({ nSplits: 5, shuffle: false });
  const folds = splitter.split(X);

  expect(folds.length).toBe(5);

  const allTest = folds.flatMap((fold) => fold.testIndices).sort((a, b) => a - b);
  expect(allTest).toEqual(X);

  for (const fold of folds) {
    expect(fold.testIndices.length).toBe(2);
    const overlap = fold.trainIndices.filter((idx) => fold.testIndices.includes(idx));
    expect(overlap.length).toBe(0);
  }
});

test("KFold shuffle is deterministic for a fixed randomState", () => {
  const X = Array.from({ length: 12 }, (_, i) => i);
  const a = new KFold({ nSplits: 4, shuffle: true, randomState: 123 }).split(X);
  const b = new KFold({ nSplits: 4, shuffle: true, randomState: 123 }).split(X);
  expect(a).toEqual(b);
});

test("StratifiedKFold preserves class balance across folds", () => {
  const X = Array.from({ length: 12 }, (_, i) => i);
  const y = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1];

  const folds = new StratifiedKFold({ nSplits: 4, shuffle: false }).split(X, y);
  expect(folds.length).toBe(4);

  for (const fold of folds) {
    const yTest = fold.testIndices.map((idx) => y[idx]);
    const positives = yTest.filter((v) => v === 1).length;
    const negatives = yTest.filter((v) => v === 0).length;
    expect(positives).toBe(1);
    expect(negatives).toBe(2);
  }
});

test("StratifiedKFold rejects nSplits larger than smallest class", () => {
  const X = Array.from({ length: 6 }, (_, i) => i);
  const y = [0, 0, 0, 0, 1, 1];

  expect(() => new StratifiedKFold({ nSplits: 3 }).split(X, y)).toThrow(
    /cannot exceed the smallest class count/i,
  );
});
