import { expect, test } from "bun:test";
import { ColumnTransformer } from "../src/pipeline/ColumnTransformer";
import { MinMaxScaler } from "../src/preprocessing/MinMaxScaler";
import { OneHotEncoder } from "../src/preprocessing/OneHotEncoder";

test("ColumnTransformer applies per-column transforms and passthrough remainder", () => {
  const X = [
    [1, 10, 100],
    [2, 20, 200],
    [3, 10, 300],
  ];

  const ct = new ColumnTransformer(
    [
      ["scale_col0", new MinMaxScaler(), [0]],
      ["encode_col1", new OneHotEncoder(), [1]],
    ],
    { remainder: "passthrough" },
  );

  const transformed = ct.fitTransform(X);
  // col0 -> 1 feature, col1 one-hot -> 2 features, remainder col2 -> 1 feature.
  expect(transformed.length).toBe(3);
  expect(transformed[0].length).toBe(4);
  expect(transformed[0]).toEqual([0, 1, 0, 100]);
  expect(transformed[1]).toEqual([0.5, 0, 1, 200]);
  expect(transformed[2]).toEqual([1, 1, 0, 300]);
});

test("ColumnTransformer transform is deterministic after fit", () => {
  const X = [
    [1, 10],
    [2, 20],
    [3, 10],
  ];
  const ct = new ColumnTransformer([
    ["scale", new MinMaxScaler(), [0]],
    ["encode", new OneHotEncoder(), [1]],
  ]);
  ct.fit(X);
  const first = ct.transform(X);
  const second = ct.transform(X);
  expect(first).toEqual(second);
});
