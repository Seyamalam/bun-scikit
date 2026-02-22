import { expect, test } from "bun:test";
import { LogisticRegression, Pipeline, StandardScaler } from "../src";

test("Pipeline matches manual scaler + logistic workflow", () => {
  const X = [[0], [1], [2], [3], [4], [5]];
  const y = [0, 0, 0, 1, 1, 1];

  const scaler = new StandardScaler();
  const XScaled = scaler.fitTransform(X);
  const manual = new LogisticRegression({
    solver: "gd",
    learningRate: 0.8,
    maxIter: 20,
    tolerance: 1e-4,
    l2: 0.01,
  });
  manual.fit(XScaled, y);
  const manualPredictions = manual.predict(XScaled);

  const pipeline = new Pipeline([
    ["scaler", new StandardScaler()],
    [
      "classifier",
      new LogisticRegression({
        solver: "gd",
        learningRate: 0.8,
        maxIter: 20,
        tolerance: 1e-4,
        l2: 0.01,
      }),
    ],
  ]);
  pipeline.fit(X, y);
  const pipelinePredictions = pipeline.predict(X);

  expect(pipelinePredictions).toEqual(manualPredictions);
});

test("Pipeline supports transformer-only fitTransform", () => {
  const X = [
    [1, 10],
    [2, 20],
    [3, 30],
  ];

  const pipeline = new Pipeline([["scaler", new StandardScaler()]]);
  const transformed = pipeline.fitTransform(X);

  expect(transformed.length).toBe(3);
  expect(transformed[0].length).toBe(2);
  expect(transformed[0][0]).toBeCloseTo(-1.22474487, 6);
  expect(transformed[2][1]).toBeCloseTo(1.22474487, 6);
});

test("Pipeline throws for duplicate step names", () => {
  expect(
    () =>
      new Pipeline([
        ["dup", new StandardScaler()],
        ["dup", new StandardScaler()],
      ]),
  ).toThrow(/must be unique/i);
});

test("Pipeline throws predict before fit", () => {
  const pipeline = new Pipeline([["scaler", new StandardScaler()]]);
  expect(() => pipeline.predict([[1], [2]])).toThrow(/has not been fitted/i);
});
