import { expect, test } from "bun:test";
import { PolynomialFeatures } from "../src/preprocessing/PolynomialFeatures";

test("PolynomialFeatures degree=2 with bias matches expected ordering", () => {
  const X = [[2, 3]];
  const poly = new PolynomialFeatures({ degree: 2, includeBias: true, interactionOnly: false });
  const transformed = poly.fitTransform(X);

  expect(transformed).toEqual([[1, 2, 3, 4, 6, 9]]);
  expect(poly.nFeaturesIn_).toBe(2);
  expect(poly.nOutputFeatures_).toBe(6);
});

test("PolynomialFeatures interactionOnly omits squared terms", () => {
  const X = [[2, 3, 4]];
  const poly = new PolynomialFeatures({ degree: 2, includeBias: false, interactionOnly: true });
  const transformed = poly.fitTransform(X);

  expect(transformed).toEqual([[2, 3, 4, 6, 8, 12]]);
});

test("PolynomialFeatures transform is deterministic after fit", () => {
  const XTrain = [
    [1, 2],
    [3, 4],
  ];
  const XTest = [[5, 6]];

  const poly = new PolynomialFeatures({ degree: 3, includeBias: false });
  poly.fit(XTrain);
  const first = poly.transform(XTest);
  const second = poly.transform(XTest);

  expect(first).toEqual(second);
});
