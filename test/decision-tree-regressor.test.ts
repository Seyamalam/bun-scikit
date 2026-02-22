import { expect, test } from "bun:test";
import { DecisionTreeRegressor } from "../src/tree/DecisionTreeRegressor";

test("DecisionTreeRegressor fits a simple nonlinear mapping", () => {
  const X = [[0], [1], [2], [3], [4], [5]];
  const y = [0, 1, 4, 9, 16, 25];

  const model = new DecisionTreeRegressor({
    maxDepth: 8,
    minSamplesSplit: 2,
    minSamplesLeaf: 1,
    randomState: 42,
  });
  model.fit(X, y);

  const predictions = model.predict(X);
  for (let i = 0; i < y.length; i += 1) {
    expect(predictions[i]).toBeCloseTo(y[i], 8);
  }
  expect(model.score(X, y)).toBeGreaterThan(0.999999);
});
