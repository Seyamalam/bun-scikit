import { expect, test } from "bun:test";
import { LogisticRegression } from "../src/linear_model/LogisticRegression";
import { getZigKernels } from "../src/native/zigKernels";

test("LogisticRegression learns a linearly separable boundary", () => {
  const X = [[0], [1], [2], [3], [4], [5]];
  const y = [0, 0, 0, 1, 1, 1];

  const model = new LogisticRegression({
    learningRate: 0.2,
    maxIter: 40_000,
    tolerance: 1e-10,
  });
  model.fit(X, y);

  const preds = model.predict(X);
  const accuracy = preds.filter((pred, idx) => pred === y[idx]).length / y.length;
  expect(accuracy).toBeGreaterThan(0.99);

  const proba = model.predictProba([[2.5]])[0][1];
  expect(proba).toBeGreaterThan(0.4);
  expect(proba).toBeLessThan(0.6);
});

test("LogisticRegression supports explicit JS backend", () => {
  const X = [[0], [1], [2], [3], [4], [5]];
  const y = [0, 0, 0, 1, 1, 1];

  const model = new LogisticRegression({
    backend: "js",
    learningRate: 0.2,
    maxIter: 40_000,
    tolerance: 1e-10,
  });

  model.fit(X, y);
  expect(model.fitBackend_).toBe("js");
  expect(model.predict([[-1], [6]])).toEqual([0, 1]);
});

test("LogisticRegression zig backend behavior is deterministic", () => {
  const X = [[0], [1], [2], [3], [4], [5]];
  const y = [0, 0, 0, 1, 1, 1];

  const kernels = getZigKernels();
  const model = new LogisticRegression({
    backend: "zig",
    learningRate: 0.2,
    maxIter: 40_000,
    tolerance: 1e-10,
  });

  if (!kernels) {
    expect(() => model.fit(X, y)).toThrow(/backend 'zig' requested/i);
    return;
  }

  model.fit(X, y);
  expect(model.fitBackend_).toBe("zig");
  expect(model.predict([[-1], [6]])).toEqual([0, 1]);
});
