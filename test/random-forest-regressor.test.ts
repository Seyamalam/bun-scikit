import { expect, test } from "bun:test";
import { RandomForestRegressor } from "../src/ensemble/RandomForestRegressor";

test("RandomForestRegressor learns a smooth regression pattern", () => {
  const X = [[0], [1], [2], [3], [4], [5], [6], [7]];
  const y = [1, 2, 5, 10, 17, 26, 37, 50];

  const model = new RandomForestRegressor({
    nEstimators: 80,
    maxDepth: 8,
    minSamplesLeaf: 1,
    bootstrap: false,
    randomState: 42,
  });
  model.fit(X, y);

  const score = model.score(X, y);
  expect(score).toBeGreaterThan(0.95);

  const trainPredictions = model.predict(X);
  let mae = 0;
  for (let i = 0; i < y.length; i += 1) {
    mae += Math.abs(trainPredictions[i] - y[i]);
  }
  mae /= y.length;
  expect(mae).toBeLessThan(2.5);
});
