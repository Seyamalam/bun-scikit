import { expect, test } from "bun:test";
import {
  EmpiricalCovariance,
  LedoitWolf,
  MinCovDet,
  OAS,
} from "../src";

const X = [
  [0.0, 0.1],
  [0.2, -0.1],
  [0.1, 0.0],
  [2.0, 2.1],
  [2.2, 1.9],
  [1.9, 2.0],
];

test("EmpiricalCovariance fits and computes Mahalanobis distances", () => {
  const model = new EmpiricalCovariance().fit(X);
  const d = model.mahalanobis(X);
  expect(d.length).toBe(X.length);
  expect(model.score(X)).toBeFinite();
});

test("LedoitWolf and OAS fit shrinkage covariance estimators", () => {
  const lw = new LedoitWolf().fit(X);
  const oas = new OAS().fit(X);
  expect(lw.shrinkage_).toBeGreaterThanOrEqual(0);
  expect(oas.shrinkage_).toBeGreaterThanOrEqual(0);
  expect(lw.score(X)).toBeFinite();
  expect(oas.score(X)).toBeFinite();
});

test("MinCovDet finds robust support and supports scoring", () => {
  const model = new MinCovDet({ supportFraction: 0.7, maxIter: 3 }).fit(X);
  expect(model.support_).not.toBeNull();
  expect(model.support_!.length).toBe(X.length);
  const d = model.mahalanobis([[0.1, 0.0], [5.0, 5.0]]);
  expect(d[1]).toBeGreaterThan(d[0]);
});
