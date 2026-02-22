import { expect, test } from "bun:test";
import { RepeatedKFold } from "../src/model_selection/RepeatedKFold";
import { RepeatedStratifiedKFold } from "../src/model_selection/RepeatedStratifiedKFold";

test("RepeatedKFold returns nRepeats * nSplits folds and covers each sample per repeat", () => {
  const X = Array.from({ length: 12 }, (_, i) => i);
  const nSplits = 3;
  const nRepeats = 2;

  const folds = new RepeatedKFold({
    nSplits,
    nRepeats,
    randomState: 5,
  }).split(X);

  expect(folds.length).toBe(nSplits * nRepeats);

  for (let repeat = 0; repeat < nRepeats; repeat += 1) {
    const chunk = folds.slice(repeat * nSplits, (repeat + 1) * nSplits);
    const allTest = chunk.flatMap((fold) => fold.testIndices).sort((a, b) => a - b);
    expect(allTest).toEqual(X);
  }
});

test("RepeatedKFold is deterministic for fixed randomState", () => {
  const X = Array.from({ length: 15 }, (_, i) => i);

  const a = new RepeatedKFold({
    nSplits: 5,
    nRepeats: 2,
    randomState: 77,
  }).split(X);
  const b = new RepeatedKFold({
    nSplits: 5,
    nRepeats: 2,
    randomState: 77,
  }).split(X);

  expect(a).toEqual(b);
});

test("RepeatedStratifiedKFold preserves class balance per fold and is deterministic", () => {
  const X = Array.from({ length: 12 }, (_, i) => i);
  const y = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1];
  const nSplits = 4;
  const nRepeats = 2;

  const splitter = new RepeatedStratifiedKFold({
    nSplits,
    nRepeats,
    randomState: 11,
  });
  const folds = splitter.split(X, y);
  const foldsAgain = splitter.split(X, y);

  expect(folds.length).toBe(nSplits * nRepeats);
  expect(folds).toEqual(foldsAgain);

  for (const fold of folds) {
    const yTest = fold.testIndices.map((idx) => y[idx]);
    const positives = yTest.filter((v) => v === 1).length;
    const negatives = yTest.filter((v) => v === 0).length;
    expect(positives).toBe(1);
    expect(negatives).toBe(2);
  }
});
