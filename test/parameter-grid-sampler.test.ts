import { expect, test } from "bun:test";
import { ParameterGrid, ParameterSampler, drawParameterSamples } from "../src";

test("ParameterGrid expands cartesian products", () => {
  const grid = new ParameterGrid({ alpha: [0.1, 1], fitIntercept: [true, false] });
  const rows = grid.toArray();
  expect(grid.length).toBe(4);
  expect(rows).toContainEqual({ alpha: 0.1, fitIntercept: true });
  expect(rows).toContainEqual({ alpha: 1, fitIntercept: false });
});

test("ParameterGrid supports list of grids", () => {
  const grid = new ParameterGrid([
    { alpha: [0.1, 1] },
    { solver: ["lbfgs"], maxIter: [100, 200] },
  ]);
  const rows = grid.toArray();
  expect(rows.length).toBe(4);
});

test("ParameterSampler is deterministic and can sample without replacement", () => {
  const distributions = { a: [1, 2], b: ["x", "y"] };
  const first = new ParameterSampler(distributions, 4, 42).toArray();
  const second = new ParameterSampler(distributions, 4, 42).toArray();
  expect(first).toEqual(second);
  expect(new Set(first.map((row) => JSON.stringify(row))).size).toBe(4);
});

test("drawParameterSamples supports repeated draws when nIter exceeds cartesian size", () => {
  const rows = drawParameterSamples({ a: [1], b: [2, 3] }, 6, 7);
  expect(rows.length).toBe(6);
});
