import { expect, test } from "bun:test";
import { LogisticRegression, OneHotEncoder, Pipeline, SimpleImputer } from "../src";

test("Pipeline composes SimpleImputer + OneHotEncoder + LogisticRegression", () => {
  const X = [[Number.NaN], [1], [2], [1], [2], [1], [2], [2]];
  const y = [0, 0, 1, 0, 1, 0, 1, 1];

  const pipeline = new Pipeline([
    ["imputer", new SimpleImputer({ strategy: "constant", fillValue: 1 })],
    ["encoder", new OneHotEncoder({ handleUnknown: "ignore" })],
    [
      "classifier",
      new LogisticRegression({
        solver: "gd",
        learningRate: 0.8,
        maxIter: 80,
        tolerance: 1e-5,
      }),
    ],
  ]);

  pipeline.fit(X, y);
  const predictions = pipeline.predict(X);
  const correct = predictions.filter((value, index) => value === y[index]).length;
  expect(correct / y.length).toBeGreaterThan(0.99);
});
