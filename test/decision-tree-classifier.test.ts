import { expect, test } from "bun:test";
import { DecisionTreeClassifier } from "../src/tree/DecisionTreeClassifier";
import { getZigKernels } from "../src/native/zigKernels";

test("DecisionTreeClassifier fits a separable threshold dataset", () => {
  const X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [2, 2],
    [2, 3],
    [3, 2],
  ];
  const y = [0, 0, 0, 1, 1, 1];

  const model = new DecisionTreeClassifier({
    maxDepth: 3,
    randomState: 42,
  });

  const previous = process.env.BUN_SCIKIT_ENABLE_ZIG;
  process.env.BUN_SCIKIT_ENABLE_ZIG = "0";
  try {
    model.fit(X, y);
  } finally {
    if (previous === undefined) {
      delete process.env.BUN_SCIKIT_ENABLE_ZIG;
    } else {
      process.env.BUN_SCIKIT_ENABLE_ZIG = previous;
    }
  }

  expect(model.predict(X)).toEqual(y);
  expect(model.score(X, y)).toBe(1);
  expect(model.fitBackend_).toBe("js");
});

test("DecisionTreeClassifier can use Zig backend when native tree symbols are available", () => {
  const previousTreeBackend = process.env.BUN_SCIKIT_TREE_BACKEND;
  process.env.BUN_SCIKIT_TREE_BACKEND = "zig";
  try {
    const kernels = getZigKernels();
    if (
      !kernels ||
      !kernels.decisionTreeModelCreate ||
      !kernels.decisionTreeModelFit ||
      !kernels.decisionTreeModelPredict ||
      !kernels.decisionTreeModelDestroy
    ) {
      return;
    }

    const X = [
      [0, 0],
      [0, 1],
      [1, 0],
      [2, 2],
      [2, 3],
      [3, 2],
    ];
    const y = [0, 0, 0, 1, 1, 1];

    const model = new DecisionTreeClassifier({
      maxDepth: 3,
      randomState: 42,
    });
    model.fit(X, y);

    expect(model.fitBackend_).toBe("zig");
    expect(model.predict(X)).toEqual(y);
    expect(model.score(X, y)).toBe(1);
  } finally {
    if (previousTreeBackend === undefined) {
      delete process.env.BUN_SCIKIT_TREE_BACKEND;
    } else {
      process.env.BUN_SCIKIT_TREE_BACKEND = previousTreeBackend;
    }
  }
});

test("DecisionTreeClassifier supports multiclass with JS backend", () => {
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

    const model = new DecisionTreeClassifier({
      maxDepth: 4,
      randomState: 42,
    });
    model.fit(X, y);

    expect(model.fitBackend_).toBe("js");
    expect(model.classes_).toEqual([0, 1, 2]);
    expect(model.predict(X)).toEqual(y);
    expect(model.score(X, y)).toBe(1);
  } finally {
    if (previousTreeBackend === undefined) {
      delete process.env.BUN_SCIKIT_TREE_BACKEND;
    } else {
      process.env.BUN_SCIKIT_TREE_BACKEND = previousTreeBackend;
    }
  }
});
