import { expect, test } from "bun:test";
import { LinearRegression } from "../src/linear_model/LinearRegression";
import { getZigKernels } from "../src/native/zigKernels";

test("LinearRegression (normal equation) fits a simple line", () => {
  const X = [[1], [2], [3], [4], [5]];
  const y = [3, 5, 7, 9, 11]; // y = 2x + 1

  const model = new LinearRegression({ solver: "normal" });
  model.fit(X, y);

  expect(model.intercept_).toBeCloseTo(1, 4);
  expect(model.coef_[0]).toBeCloseTo(2, 4);

  const prediction = model.predict([[6]])[0];
  expect(prediction).toBeCloseTo(13, 4);
});

test("LinearRegression zig backend behavior is deterministic", () => {
  const X = [[1], [2], [3], [4], [5]];
  const y = [3, 5, 7, 9, 11];

  const kernels = getZigKernels();
  const model = new LinearRegression({ solver: "normal" });

  if (!kernels) {
    expect(() => model.fit(X, y)).toThrow(/requires native zig kernels/i);
    return;
  }

  model.fit(X, y);
  expect(model.fitBackend_).toBe("zig");
  expect(model.predict([[6]])[0]).toBeCloseTo(13, 4);
});
