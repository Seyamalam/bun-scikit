import { expect, test } from "bun:test";
import {
  BernoulliNB,
  CategoricalNB,
  ComplementNB,
  MultinomialNB,
} from "../src";

test("naive-bayes parity classes fit and predict", () => {
  const XCounts = [
    [2, 1, 0],
    [3, 0, 1],
    [0, 2, 3],
    [1, 3, 2],
  ];
  const y = [0, 0, 1, 1];

  const multinomial = new MultinomialNB().fit(XCounts, y);
  expect(multinomial.predict(XCounts).length).toBe(y.length);
  expect(multinomial.predictProba(XCounts).length).toBe(y.length);

  const complement = new ComplementNB().fit(XCounts, y);
  expect(complement.predict(XCounts).length).toBe(y.length);
  expect(complement.predictProba(XCounts).length).toBe(y.length);

  const bernoulli = new BernoulliNB({ binarize: 0 }).fit(XCounts, y);
  expect(bernoulli.predict(XCounts).length).toBe(y.length);
  expect(bernoulli.predictProba(XCounts).length).toBe(y.length);
});

test("CategoricalNB handles integer-encoded categories", () => {
  const X = [
    [0, 1],
    [1, 0],
    [2, 2],
    [2, 1],
    [1, 2],
    [0, 0],
  ];
  const y = [0, 0, 1, 1, 1, 0];

  const model = new CategoricalNB().fit(X, y);
  expect(model.predict(X).length).toBe(X.length);
  expect(model.predictProba(X).length).toBe(X.length);
});

