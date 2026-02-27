import { expect, test } from "bun:test";
import {
  GenericUnivariateSelect,
  SelectFdr,
  SelectFpr,
  SelectFwe,
  SequentialFeatureSelector,
  f_classif,
} from "../src";
import { LogisticRegression } from "../src";

test("SelectFpr/Fdr/Fwe select informative feature", () => {
  const X = [
    [-3, 0, 10],
    [-2, 0, 10],
    [-1, 0, 10],
    [1, 0, 10],
    [2, 0, 10],
    [3, 0, 10],
  ];
  const y = [0, 0, 0, 1, 1, 1];

  const fpr = new SelectFpr({ scoreFunc: f_classif, alpha: 0.2 }).fit(X, y);
  const fdr = new SelectFdr({ scoreFunc: f_classif, alpha: 0.2 }).fit(X, y);
  const fwe = new SelectFwe({ scoreFunc: f_classif, alpha: 0.2 }).fit(X, y);

  expect((fpr.getSupport(true) as number[]).includes(0)).toBe(true);
  expect((fdr.getSupport(true) as number[]).includes(0)).toBe(true);
  expect((fwe.getSupport(true) as number[]).includes(0)).toBe(true);
});

test("GenericUnivariateSelect supports multiple modes", () => {
  const X = [
    [-3, 0, 10],
    [-2, 0, 10],
    [-1, 0, 10],
    [1, 0, 10],
    [2, 0, 10],
    [3, 0, 10],
  ];
  const y = [0, 0, 0, 1, 1, 1];

  const generic = new GenericUnivariateSelect({
    scoreFunc: f_classif,
    mode: "k_best",
    param: 1,
  }).fit(X, y);
  expect(generic.getSupport(true)).toEqual([0]);
});

test("SequentialFeatureSelector supports forward and backward search", () => {
  const X = [
    [-3, 0, 10],
    [-2, 0, 10],
    [-1, 0, 10],
    [1, 0, 10],
    [2, 0, 10],
    [3, 0, 10],
    [4, 0, 10],
    [5, 0, 10],
  ];
  const y = [0, 0, 0, 1, 1, 1, 1, 1];

  const forward = new SequentialFeatureSelector(
    () =>
      new LogisticRegression({
        maxIter: 300,
        learningRate: 0.1,
        tolerance: 1e-6,
      }),
    {
      nFeaturesToSelect: 1,
      direction: "forward",
      cv: 3,
    },
  ).fit(X, y);
  expect(forward.getSupport(true)).toEqual([0]);

  const backward = new SequentialFeatureSelector(
    () =>
      new LogisticRegression({
        maxIter: 300,
        learningRate: 0.1,
        tolerance: 1e-6,
      }),
    {
      nFeaturesToSelect: 1,
      direction: "backward",
      cv: 3,
    },
  ).fit(X, y);
  expect(backward.getSupport(true)).toEqual([0]);
});
