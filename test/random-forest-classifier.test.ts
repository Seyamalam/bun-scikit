import { expect, test } from "bun:test";
import { RandomForestClassifier } from "../src/ensemble/RandomForestClassifier";

test("RandomForestClassifier learns separable clusters", () => {
  const X = [
    [0.0, 0.1],
    [0.2, 0.1],
    [0.1, 0.2],
    [1.0, 1.0],
    [1.1, 0.9],
    [0.9, 1.1],
  ];
  const y = [0, 0, 0, 1, 1, 1];

  const model = new RandomForestClassifier({
    nEstimators: 25,
    maxDepth: 4,
    randomState: 42,
  });
  model.fit(X, y);

  const predictions = model.predict(X);
  expect(predictions).toEqual(y);
  expect(model.score(X, y)).toBe(1);
});

test("RandomForestClassifier supports multiclass and falls back to JS voting", () => {
  const previousTreeBackend = process.env.BUN_SCIKIT_TREE_BACKEND;
  process.env.BUN_SCIKIT_TREE_BACKEND = "js";
  try {
    const X = [
      [0.0, 0.1],
      [0.1, 0.2],
      [0.2, 0.0],
      [2.0, 2.1],
      [2.2, 1.9],
      [1.8, 2.2],
      [4.0, 4.1],
      [4.1, 3.9],
      [3.9, 4.2],
    ];
    const y = [0, 0, 0, 1, 1, 1, 2, 2, 2];

    const model = new RandomForestClassifier({
      nEstimators: 40,
      maxDepth: 4,
      randomState: 42,
    });
    model.fit(X, y);

    expect(model.fitBackend_).toBe("js");
    expect(model.classes_).toEqual([0, 1, 2]);
    expect(model.score(X, y)).toBeGreaterThan(0.95);
  } finally {
    if (previousTreeBackend === undefined) {
      delete process.env.BUN_SCIKIT_TREE_BACKEND;
    } else {
      process.env.BUN_SCIKIT_TREE_BACKEND = previousTreeBackend;
    }
  }
});
