import { expect, test } from "bun:test";
import {
  KBinsDiscretizer,
  PowerTransformer,
  QuantileTransformer,
} from "../src";

test("QuantileTransformer supports normal output and inverse transform", () => {
  const X = [[0], [1], [2], [3], [4], [5]];
  const qt = new QuantileTransformer({ nQuantiles: 6, outputDistribution: "normal" }).fit(X);
  const transformed = qt.transform(X);
  expect(transformed.length).toBe(X.length);
  const restored = qt.inverseTransform(transformed);
  expect(restored.length).toBe(X.length);
  expect(Math.abs(restored[2][0] - X[2][0])).toBeLessThan(0.2);
});

test("PowerTransformer transforms and inverts positive values", () => {
  const X = [[1], [2], [3], [4], [5]];
  const pt = new PowerTransformer({ method: "box-cox", standardize: true }).fit(X);
  const transformed = pt.transform(X);
  expect(transformed.length).toBe(X.length);
  const restored = pt.inverseTransform(transformed);
  expect(Math.abs(restored[3][0] - X[3][0])).toBeLessThan(0.2);
});

test("KBinsDiscretizer supports ordinal and onehot encodings", () => {
  const X = [[0], [1], [2], [3], [4], [5]];
  const kbOrdinal = new KBinsDiscretizer({
    nBins: 3,
    encode: "ordinal",
    strategy: "quantile",
  }).fit(X);
  const ordinal = kbOrdinal.transform(X);
  expect(ordinal[0][0]).toBe(0);
  expect(ordinal[5][0]).toBe(2);

  const kbOneHot = new KBinsDiscretizer({
    nBins: 3,
    encode: "onehot-dense",
    strategy: "uniform",
  }).fit(X);
  const oneHot = kbOneHot.transform(X);
  expect(oneHot[0].length).toBe(3);
  expect(oneHot[0].reduce((a, b) => a + b, 0)).toBe(1);
});
