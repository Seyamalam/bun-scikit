import { expect, test } from "bun:test";
import {
  ARDRegression,
  BayesianRidge,
  GammaRegressor,
  HuberRegressor,
  LogisticRegressionCV,
  MultiTaskElasticNet,
  MultiTaskElasticNetCV,
  MultiTaskLasso,
  MultiTaskLassoCV,
  PassiveAggressiveClassifier,
  PassiveAggressiveRegressor,
  Perceptron,
  PoissonRegressor,
  QuantileRegressor,
} from "../src";

test("new linear-model parity estimators fit and predict", () => {
  const X = [[0], [1], [2], [3], [4], [5]];
  const y = [0, 1, 2, 3, 4, 5];
  const yPositive = [1, 2, 3, 4, 5, 6];
  const yClass = [0, 0, 0, 1, 1, 1];

  const bayes = new BayesianRidge().fit(X, y);
  expect(bayes.predict(X).length).toBe(X.length);

  const ard = new ARDRegression().fit(X, y);
  expect(ard.predict(X).length).toBe(X.length);

  const perceptron = new Perceptron({ maxIter: 50, randomState: 3 }).fit(X, yClass);
  expect(perceptron.predict(X).length).toBe(X.length);

  const paCls = new PassiveAggressiveClassifier({ maxIter: 50 }).fit(X, yClass);
  expect(paCls.predict(X).length).toBe(X.length);

  const paReg = new PassiveAggressiveRegressor({ maxIter: 50 }).fit(X, y);
  expect(paReg.predict(X).length).toBe(X.length);

  const huber = new HuberRegressor({ maxIter: 100 }).fit(X, y);
  expect(huber.predict(X).length).toBe(X.length);

  const logCv = new LogisticRegressionCV({ cv: 3, maxIter: 120 }).fit(X, yClass);
  expect(logCv.predict(X).length).toBe(X.length);

  const poisson = new PoissonRegressor({ maxIter: 120 }).fit(X, yPositive);
  expect(poisson.predict(X).length).toBe(X.length);

  const gamma = new GammaRegressor({ maxIter: 120 }).fit(X, yPositive);
  expect(gamma.predict(X).length).toBe(X.length);

  const quantile = new QuantileRegressor({ maxIter: 120 }).fit(X, y);
  expect(quantile.predict(X).length).toBe(X.length);
});

test("multi-task linear models fit matrix targets", () => {
  const X = [[0], [1], [2], [3], [4], [5]];
  const Y = [
    [0, 0],
    [1, 2],
    [2, 4],
    [3, 6],
    [4, 8],
    [5, 10],
  ];

  const lasso = new MultiTaskLasso({ maxIter: 200 }).fit(X, Y);
  expect(lasso.predict(X)[0].length).toBe(2);

  const enet = new MultiTaskElasticNet({ maxIter: 200 }).fit(X, Y);
  expect(enet.predict(X)[0].length).toBe(2);

  const lassoCv = new MultiTaskLassoCV({ cv: 3, maxIter: 120 }).fit(X, Y);
  expect(lassoCv.predict(X).length).toBe(X.length);

  const enetCv = new MultiTaskElasticNetCV({ cv: 3, maxIter: 120 }).fit(X, Y);
  expect(enetCv.predict(X).length).toBe(X.length);
});

