import { expect, test } from "bun:test";
import { LinearRegression } from "../src/linear_model/LinearRegression";
import { RandomizedSearchCV } from "../src/model_selection/RandomizedSearchCV";

test("RandomizedSearchCV runs deterministic search with fixed randomState", () => {
  const X = [[0], [1], [2], [3], [4], [5], [6], [7]];
  const y = [1, 3, 5, 7, 9, 11, 13, 15];

  const searchA = new RandomizedSearchCV(
    (params) =>
      new LinearRegression({
        solver: "normal",
        fitIntercept: Boolean(params.fitIntercept),
      }),
    { fitIntercept: [false, true] },
    { cv: 4, scoring: "r2", refit: true, nIter: 6, randomState: 11 },
  );
  const searchB = new RandomizedSearchCV(
    (params) =>
      new LinearRegression({
        solver: "normal",
        fitIntercept: Boolean(params.fitIntercept),
      }),
    { fitIntercept: [false, true] },
    { cv: 4, scoring: "r2", refit: true, nIter: 6, randomState: 11 },
  );

  searchA.fit(X, y);
  searchB.fit(X, y);

  expect(searchA.cvResults_.map((row) => row.params)).toEqual(
    searchB.cvResults_.map((row) => row.params),
  );
  expect(searchA.bestParams_).toEqual(searchB.bestParams_);
  expect(searchA.predict([[8]])[0]).toBeCloseTo(17, 6);
});
