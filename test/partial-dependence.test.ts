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
  expect(result.values[0].length).toBe(6);
  expect(result.individual).toBeDefined();

  const avg0 = result.average[0];
  expect(avg0[avg0.length - 1]).toBeGreaterThan(avg0[0]);

  const avg1 = result.average[1];
  expect(Math.abs(avg1[avg1.length - 1] - avg1[0])).toBeLessThan(1e-6);
});
