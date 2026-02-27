import { expect, test } from "bun:test";
import { LinearRegression, partialDependence } from "../src";

test("partialDependence captures monotonic effect of informative feature", () => {
  const X = [
    [0, 10],
    [1, 10],
    [2, 10],
    [3, 10],
    [4, 10],
    [5, 10],
  ];
  const y = [0, 2, 4, 6, 8, 10];
  const estimator = new LinearRegression().fit(X, y);

  const result = partialDependence(estimator, X, {
    features: [0, 1],
    gridResolution: 6,
    responseMethod: "predict",
    kind: "both",
  });

  expect(result.average.length).toBe(2);
  expect((result.values[0] as number[]).length).toBe(6);
  expect(result.individual).toBeDefined();
  expect(result.responseMethodUsed).toBe("predict");
  expect(result.deciles["0"].length).toBe(9);

  const avg0 = result.average[0] as number[];
  expect(avg0[avg0.length - 1]).toBeGreaterThan(avg0[0]);

  const avg1 = result.average[1] as number[];
  expect(Math.abs(avg1[avg1.length - 1] - avg1[0])).toBeLessThan(1e-6);
});

test("partialDependence supports 2D feature pairs", () => {
  const X = [
    [0, 0, 10],
    [1, 0, 10],
    [2, 1, 10],
    [3, 1, 10],
    [4, 0, 10],
    [5, 1, 10],
  ];
  const y = [0, 1, 3, 4, 4, 6];
  const estimator = new LinearRegression().fit(X, y);

  const result = partialDependence(estimator, X, {
    features: [[0, 1]],
    gridResolution: 4,
    kind: "both",
    responseMethod: "predict",
  });

  expect(result.features.length).toBe(1);
  const grid = result.values[0] as [number[], number[]];
  expect(grid[0].length).toBe(4);
  expect(grid[1].length).toBe(4);
  const avg = result.average[0] as number[][];
  expect(avg.length).toBe(4);
  expect(avg[0].length).toBe(4);
  expect(result.individual).toBeDefined();
});
