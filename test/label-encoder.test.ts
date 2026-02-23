import { expect, test } from "bun:test";
import { LabelEncoder } from "../src/preprocessing/LabelEncoder";

test("LabelEncoder encodes sorted unique classes to [0..k-1]", () => {
  const y = [3, 1, 3, 2, 1];
  const encoder = new LabelEncoder();

  const encoded = encoder.fitTransform(y);
  expect(encoder.classes_).toEqual([1, 2, 3]);
  expect(encoded).toEqual([2, 0, 2, 1, 0]);
});

test("LabelEncoder inverseTransform reconstructs labels", () => {
  const encoder = new LabelEncoder().fit([10, 20, 10, 30]);
  const decoded = encoder.inverseTransform([2, 1, 0, 2]);
  expect(decoded).toEqual([30, 20, 10, 30]);
});
