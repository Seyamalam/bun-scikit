import { expect, test } from "bun:test";
import { DummyClassifier, GaussianNB, KNeighborsClassifier, VotingClassifier } from "../src";

test("VotingClassifier hard voting combines base estimators", () => {
  const X = [[0], [1], [2], [3], [4], [5], [6], [7]];
  const y = [0, 0, 0, 0, 1, 1, 1, 1];

  const voting = new VotingClassifier(
    [
      ["gnb", () => new GaussianNB()],
      ["dummy", () => new DummyClassifier({ strategy: "prior" })],
    ],
    { voting: "hard", weights: [2, 1] },
  );
  voting.fit(X, y);
  const preds = voting.predict(X);

  expect(preds.length).toBe(X.length);
  expect(voting.score(X, y)).toBeGreaterThan(0.85);
});

test("VotingClassifier soft voting averages probabilities", () => {
  const X = [[0], [1], [2], [3], [4], [5]];
  const y = [0, 0, 0, 1, 1, 1];

  const voting = new VotingClassifier(
    [
      ["gnb", () => new GaussianNB()],
      ["dummy", () => new DummyClassifier({ strategy: "prior" })],
    ],
    { voting: "soft" },
  ).fit(X, y);

  const proba = voting.predictProba(X);
  expect(proba.length).toBe(X.length);
  expect(proba[0].length).toBe(2);
  for (let i = 0; i < proba.length; i += 1) {
    expect(proba[i][0] + proba[i][1]).toBeCloseTo(1, 10);
  }
});

test("VotingClassifier predictProba requires predictProba on all estimators", () => {
  const X = [[0], [1], [2], [3]];
  const y = [0, 0, 1, 1];
  const voting = new VotingClassifier([
    ["knn", () => new KNeighborsClassifier({ nNeighbors: 1 })],
    ["dummy", () => new DummyClassifier({ strategy: "prior" })],
  ]).fit(X, y);

  expect(() => voting.predictProba(X)).toThrow(/requires all estimators to implement predictproba/i);
});
