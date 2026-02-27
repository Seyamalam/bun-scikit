import { expect, test } from "bun:test";
import { GroupKFold } from "../src/model_selection/GroupKFold";

test("GroupKFold keeps group members in the same fold", () => {
  const X = Array.from({ length: 12 }, (_, i) => i);
  const groups = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5];
  const folds = new GroupKFold({ nSplits: 3 }).split(X, undefined, groups);

  expect(folds.length).toBe(3);
  for (const fold of folds) {
    const inTest = new Set(fold.testIndices);
    for (let i = 0; i < groups.length; i += 1) {
      for (let j = i + 1; j < groups.length; j += 1) {
        if (groups[i] === groups[j]) {
          expect(inTest.has(i)).toBe(inTest.has(j));
        }
      }
    }
  }
});

test("GroupKFold rejects nSplits larger than unique groups", () => {
  const X = [0, 1, 2, 3];
  const groups = [10, 10, 11, 11];
  expect(() => new GroupKFold({ nSplits: 3 }).split(X, undefined, groups)).toThrow(
    /cannot exceed unique group count/i,
  );
});
