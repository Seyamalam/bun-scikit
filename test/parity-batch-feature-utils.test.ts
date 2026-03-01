import { expect, test } from "bun:test";
import {
  DictVectorizer,
  FeatureHasher,
  FunctionTransformer,
  KernelCenterer,
  LabelBinarizer,
  MultiLabelBinarizer,
} from "../src";

test("DictVectorizer and FeatureHasher transform feature dictionaries", () => {
  const samples = [
    { city: "nyc", clicks: 3, premium: true },
    { city: "sf", clicks: 1, premium: false },
    { city: "nyc", clicks: 2, premium: true },
  ];

  const vectorizer = new DictVectorizer().fit(samples);
  const matrix = vectorizer.transform(samples);
  expect(matrix.length).toBe(samples.length);
  expect(vectorizer.getFeatureNamesOut().length).toBeGreaterThan(0);

  const hasher = new FeatureHasher({ nFeatures: 16, inputType: "dict" });
  const hashed = hasher.transform(samples);
  expect(hashed.length).toBe(samples.length);
  expect(hashed[0].length).toBe(16);
});

test("FunctionTransformer and KernelCenterer apply transforms", () => {
  const X = [
    [1, 2],
    [3, 4],
  ];
  const transformer = new FunctionTransformer({
    func: (input) => input.map((row) => row.map((value) => value * 2)),
    inverseFunc: (input) => input.map((row) => row.map((value) => value / 2)),
    validate: true,
  }).fit(X);

  const transformed = transformer.transform(X);
  expect(transformed[0][0]).toBe(2);
  const inverted = transformer.inverseTransform(transformed);
  expect(inverted[1][1]).toBe(4);

  const K = [
    [1, 0.5],
    [0.5, 1],
  ];
  const centerer = new KernelCenterer().fit(K);
  const centered = centerer.transform(K);
  expect(centered.length).toBe(2);
  expect(centered[0].length).toBe(2);
});

test("LabelBinarizer and MultiLabelBinarizer encode labels", () => {
  const lb = new LabelBinarizer().fit([0, 1, 0, 1]);
  const encoded = lb.transform([0, 1, 1, 0]);
  expect(encoded.length).toBe(4);
  expect(lb.inverseTransform(encoded)).toEqual([0, 1, 1, 0]);

  const mlb = new MultiLabelBinarizer().fit([
    [1, 2],
    [2, 3],
    [1],
  ]);
  const transformed = mlb.transform([
    [1, 3],
    [2],
  ]);
  expect(transformed.length).toBe(2);
  expect(mlb.inverseTransform(transformed).length).toBe(2);
});

