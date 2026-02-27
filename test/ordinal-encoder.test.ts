import { expect, test } from "bun:test";
import { OrdinalEncoder } from "../src/preprocessing/OrdinalEncoder";

test("OrdinalEncoder fits and transforms numeric categories", () => {
  const X = [
    [2, 10],
    [1, 20],
    [2, 20],
  ];
  const encoder = new OrdinalEncoder();
  const transformed = encoder.fitTransform(X);

  expect(encoder.categories_).toEqual([[1, 2], [10, 20]]);
  expect(transformed).toEqual([
    [1, 0],
    [0, 1],
    [1, 1],
  ]);
  expect(encoder.inverseTransform(transformed)).toEqual(X);
});

test("OrdinalEncoder can encode unknown values with configured fallback", () => {
  const encoder = new OrdinalEncoder({ handleUnknown: "use_encoded_value", unknownValue: -1 })
    .fit([
      [1, 10],
      [2, 20],
    ]);

  const transformed = encoder.transform([
    [1, 999],
    [3, 10],
  ]);
  expect(transformed).toEqual([
    [0, -1],
    [-1, 0],
  ]);
});
