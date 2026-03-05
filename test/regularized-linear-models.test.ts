import { expect, test } from "bun:test";
import {
  ElasticNet,
  ElasticNetCV,
  Lasso,
  LassoCV,
  Ridge,
  RidgeClassifier,
  RidgeClassifierCV,
  RidgeCV,
} from "../src";

test("Ridge/Lasso/ElasticNet fit a simple regression signal", () => {
  const X = [[0], [1], [2], [3], [4], [5]];
  const y = [0, 1, 2, 3, 4, 5];

  const ridge = new Ridge({ alpha: 1 }).fit(X, y);
  const lasso = new Lasso({ alpha: 0.01, maxIter: 2000, tolerance: 1e-8 }).fit(X, y);
  const en = new ElasticNet({ alpha: 0.01, l1Ratio: 0.5, maxIter: 2000, tolerance: 1e-8 }).fit(X, y);

  expect(ridge.score(X, y)).toBeGreaterThan(0.98);
  expect(lasso.score(X, y)).toBeGreaterThan(0.98);
  expect(en.score(X, y)).toBeGreaterThan(0.98);
});

test("RidgeCV/LassoCV/ElasticNetCV select hyperparameters and predict", () => {
  const X = [[0], [1], [2], [3], [4], [5], [6], [7]];
  const y = [0, 1.1, 1.9, 3.1, 4.1, 4.9, 6.2, 7.1];

  const ridgeCv = new RidgeCV({ alphas: [0.01, 0.1, 1], cv: 4, randomState: 7 }).fit(X, y);
  const lassoCv = new LassoCV({
    alphas: [0.001, 0.01, 0.1],
    cv: 4,
    maxIter: 2000,
    tolerance: 1e-8,
    randomState: 7,
  }).fit(X, y);
  const enCv = new ElasticNetCV({
    alphas: [0.001, 0.01, 0.1],
    l1Ratio: [0.2, 0.5, 0.8],
    cv: 4,
    maxIter: 2000,
    tolerance: 1e-8,
    randomState: 7,
  }).fit(X, y);

  expect(ridgeCv.alpha_).toBeFinite();
  expect(lassoCv.alpha_).toBeFinite();
  expect(enCv.alpha_).toBeFinite();
  expect(enCv.l1Ratio_).toBeFinite();

  expect(ridgeCv.predict([[2.5]])[0]).toBeGreaterThan(2);
  expect(lassoCv.predict([[2.5]])[0]).toBeGreaterThan(2);
  expect(enCv.predict([[2.5]])[0]).toBeGreaterThan(2);
});

test("RidgeClassifier and RidgeClassifierCV classify a separable dataset", () => {
  const X = [[-3], [-2], [-1], [1], [2], [3], [4], [5]];
  const y = [0, 0, 0, 1, 1, 1, 1, 1];

  const ridgeClassifier = new RidgeClassifier({ alpha: 1 }).fit(X, y);
  const ridgeClassifierCv = new RidgeClassifierCV({
    alphas: [0.01, 0.1, 1, 10],
    cv: 3,
    randomState: 7,
  }).fit(X, y);

  expect(ridgeClassifier.score(X, y)).toBeGreaterThan(0.99);
  expect(ridgeClassifierCv.score(X, y)).toBeGreaterThan(0.99);
  expect(ridgeClassifier.predict([[-0.5], [2.5]])).toEqual([0, 1]);
  expect(ridgeClassifierCv.predict([[-0.5], [2.5]])).toEqual([0, 1]);
  expect(ridgeClassifierCv.alpha_).toBeFinite();
  expect(ridgeClassifierCv.cvScores_).toHaveLength(4);
});
