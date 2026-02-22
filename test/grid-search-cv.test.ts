import { expect, test } from "bun:test";
import { LinearRegression } from "../src/linear_model/LinearRegression";
import { GridSearchCV } from "../src/model_selection/GridSearchCV";

test("GridSearchCV selects parameters with best cross-validated score", () => {
  const X = [[0], [1], [2], [3], [4], [5], [6], [7]];
  const y = [1, 3, 5, 7, 9, 11, 13, 15];

  const search = new GridSearchCV(
    (params) =>
      new LinearRegression({
        solver: "normal",
        fitIntercept: Boolean(params.fitIntercept),
      }),
    { fitIntercept: [false, true] },
    { cv: 4, scoring: "r2", refit: true },
  );

  search.fit(X, y);

  expect(search.bestParams_).toEqual({ fitIntercept: true });
  expect(search.bestScore_).not.toBeNull();
  expect(search.bestScore_!).toBeGreaterThan(0.99);
  expect(search.cvResults_.length).toBe(2);

  const prediction = search.predict([[8]])[0];
  expect(prediction).toBeCloseTo(17, 6);
});

test("GridSearchCV supports errorScore fallback", () => {
  class SometimesFailEstimator {
    constructor(private readonly shouldFail: boolean) {}

    fit(): this {
      if (this.shouldFail) {
        throw new Error("intentional fit failure");
      }
      return this;
    }

    predict(X: number[][]): number[] {
      return new Array(X.length).fill(0);
    }

    score(): number {
      return 0.8;
    }
  }

  const X = [[1], [2], [3], [4], [5], [6]];
  const y = [0, 0, 0, 1, 1, 1];

  const search = new GridSearchCV(
    (params) => new SometimesFailEstimator(Boolean(params.fail)),
    { fail: [true, false] },
    { cv: 3, errorScore: -1, refit: false },
  );
  search.fit(X, y);

  expect(search.cvResults_.length).toBe(2);
  const failedRow = search.cvResults_.find((row) => row.params.fail === true);
  const successRow = search.cvResults_.find((row) => row.params.fail === false);

  expect(failedRow?.status).toBe("error");
  expect(failedRow?.meanTestScore).toBe(-1);
  expect(successRow?.status).toBe("ok");
  expect(search.bestParams_).toEqual({ fail: false });
});

test("GridSearchCV is deterministic when scores tie", () => {
  class TieEstimator {
    fit(): this {
      return this;
    }
    predict(X: number[][]): number[] {
      return new Array(X.length).fill(0);
    }
    score(): number {
      return 0.5;
    }
  }

  const X = [[1], [2], [3], [4]];
  const y = [0, 0, 1, 1];
  const search = new GridSearchCV(
    () => new TieEstimator(),
    { alpha: [0.1, 1.0, 10.0] },
    { cv: 2, refit: false },
  );
  search.fit(X, y);

  expect(search.bestParams_).toEqual({ alpha: 0.1 });
  expect(search.cvResults_.find((row) => row.params.alpha === 0.1)?.rank).toBe(1);
});
