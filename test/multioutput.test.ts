import { expect, test } from "bun:test";
import {
  ClassifierChain,
  DecisionTreeClassifier,
  LinearRegression,
  MultiOutputClassifier,
  MultiOutputRegressor,
  RegressorChain,
} from "../src";

test("MultiOutputClassifier fits and predicts matrix targets", () => {
  const X = [[0], [1], [2], [3], [4], [5]];
  const Y = [
    [0, 1],
    [0, 1],
    [0, 0],
    [1, 0],
    [1, 1],
    [1, 1],
  ];

  const model = new MultiOutputClassifier(() => new DecisionTreeClassifier({ maxDepth: 3, randomState: 7 }));
  model.fit(X, Y);
  const pred = model.predict(X);
  expect(pred.length).toBe(X.length);
  expect(pred[0].length).toBe(2);
  expect(model.score(X, Y)).toBeGreaterThan(0.8);
});

test("ClassifierChain supports chained multi-output classification", () => {
  const X = [[0], [1], [2], [3], [4], [5]];
  const Y = [
    [0, 0],
    [0, 0],
    [0, 1],
    [1, 1],
    [1, 1],
    [1, 1],
  ];

  const chain = new ClassifierChain(
    () => new DecisionTreeClassifier({ maxDepth: 3, randomState: 11 }),
    { order: [1, 0] },
  ).fit(X, Y);
  const pred = chain.predict(X);
  expect(pred.length).toBe(X.length);
  expect(pred[0].length).toBe(2);
  expect(chain.predictProba(X)[0].length).toBe(2);
});

test("MultiOutputRegressor and RegressorChain fit/predict", () => {
  const X = [[0], [1], [2], [3], [4], [5]];
  const Y = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6],
  ];

  const mo = new MultiOutputRegressor(() => new LinearRegression({ solver: "normal" })).fit(X, Y);
  const predMo = mo.predict(X);
  expect(predMo.length).toBe(X.length);
  expect(predMo[0].length).toBe(2);
  expect(mo.score(X, Y)).toBeGreaterThan(0.99);

  const chain = new RegressorChain(() => new LinearRegression({ solver: "normal" })).fit(X, Y);
  const predChain = chain.predict(X);
  expect(predChain.length).toBe(X.length);
  expect(predChain[0].length).toBe(2);
  expect(chain.score(X, Y)).toBeGreaterThan(0.99);
});
