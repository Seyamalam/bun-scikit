import { expect, test } from "bun:test";
import { meanSquaredError, r2Score } from "../src";

test("regression metrics support sampleWeight and raw multioutput values", () => {
  const yTrue = [
    [1, 10],
    [2, 20],
    [3, 30],
    [4, 40],
  ];
  const yPred = [
    [1.2, 9],
    [2.2, 19],
    [3.2, 29],
    [3.8, 41],
  ];
  const sampleWeight = [1, 2, 1, 0.5];

  const mseRaw = meanSquaredError(yTrue, yPred, {
    sampleWeight,
    multioutput: "raw_values",
  }) as number[];
  expect(mseRaw.length).toBe(2);
  expect(mseRaw[0]).toBeGreaterThan(0);
  expect(mseRaw[1]).toBeGreaterThan(0);

  const r2 = r2Score(yTrue, yPred, {
    sampleWeight,
    multioutput: "variance_weighted",
  }) as number;
  expect(r2).toBeLessThanOrEqual(1);
  expect(r2).toBeGreaterThan(0.9);
});
