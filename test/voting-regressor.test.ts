import { expect, test } from "bun:test";
import { DecisionTreeRegressor, DummyRegressor, VotingRegressor } from "../src";

test("VotingRegressor averages base regressor predictions", () => {
  const X = [[0], [1], [2], [3], [4], [5], [6], [7]];
  const y = [1, 3, 5, 7, 9, 11, 13, 15];

  const model = new VotingRegressor(
    [
      [
        "tree",
        () =>
          new DecisionTreeRegressor({
            maxDepth: 5,
            minSamplesLeaf: 1,
            randomState: 42,
          }),
      ],
      ["dummy", () => new DummyRegressor({ strategy: "mean" })],
    ],
    { weights: [3, 1] },
  ).fit(X, y);

  const preds = model.predict(X);
  expect(preds.length).toBe(X.length);
  expect(model.score(X, y)).toBeGreaterThan(0.9);
});

test("VotingRegressor validates estimator names", () => {
  const X = [[0], [1], [2], [3]];
  const y = [1, 3, 5, 7];

  const model = new VotingRegressor([
    ["dup", () => new DummyRegressor({ strategy: "mean" })],
    ["dup", () => new DummyRegressor({ strategy: "constant", constant: 3 })],
  ]);

  expect(() => model.fit(X, y)).toThrow(/must be unique/i);
});

test("VotingRegressor throws before fit", () => {
  const model = new VotingRegressor([["dummy", () => new DummyRegressor({ strategy: "mean" })]]);
  expect(() => model.predict([[0], [1]])).toThrow(/has not been fitted/i);
});
