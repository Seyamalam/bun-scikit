import { expect, test } from "bun:test";
import { DecisionTreeClassifier } from "../src/tree/DecisionTreeClassifier";
import { RandomForestClassifier } from "../src/ensemble/RandomForestClassifier";
import { getZigKernels } from "../src/native/zigKernels";

const ENFORCE_ZIG_BACKEND = process.env.BUN_SCIKIT_REQUIRE_ZIG_BACKEND === "1";

test("zig backend guard enforces native tree and forest fit paths", () => {
  if (!ENFORCE_ZIG_BACKEND) {
    return;
  }

  const kernels = getZigKernels();
  expect(kernels).toBeTruthy();
  expect(kernels?.decisionTreeModelCreate).toBeTruthy();
  expect(kernels?.decisionTreeModelFit).toBeTruthy();
  expect(kernels?.decisionTreeModelPredict).toBeTruthy();
  expect(kernels?.randomForestClassifierModelCreate).toBeTruthy();
  expect(kernels?.randomForestClassifierModelFit).toBeTruthy();
  expect(kernels?.randomForestClassifierModelPredict).toBeTruthy();

  const X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [2, 2],
    [2, 3],
    [3, 2],
  ];
  const y = [0, 0, 0, 1, 1, 1];
  const yMulti = [0, 0, 1, 1, 2, 2];

  const previousTreeBackend = process.env.BUN_SCIKIT_TREE_BACKEND;
  process.env.BUN_SCIKIT_TREE_BACKEND = "zig";
  try {
    const tree = new DecisionTreeClassifier({ maxDepth: 3, randomState: 42 });
    tree.fit(X, y);
    expect(tree.fitBackend_).toBe("zig");
    expect(tree.fitBackendLibrary_).toBeTruthy();
    tree.dispose();

    const forest = new RandomForestClassifier({
      nEstimators: 25,
      maxDepth: 4,
      randomState: 42,
    });
    forest.fit(X, y);
    expect(forest.fitBackend_).toBe("zig");
    expect(forest.fitBackendLibrary_).toBeTruthy();
    forest.dispose();

    const multiTree = new DecisionTreeClassifier({ maxDepth: 3, randomState: 42 });
    multiTree.fit(X, yMulti);
    expect(multiTree.fitBackend_).toBe("zig");
    expect(multiTree.predict(X).length).toBe(X.length);
    multiTree.dispose();

    const multiForest = new RandomForestClassifier({
      nEstimators: 20,
      maxDepth: 4,
      randomState: 42,
    });
    multiForest.fit(X, yMulti);
    expect(multiForest.fitBackend_).toBe("zig");
    expect(multiForest.predict(X).length).toBe(X.length);
    multiForest.dispose();
  } finally {
    if (previousTreeBackend === undefined) {
      delete process.env.BUN_SCIKIT_TREE_BACKEND;
    } else {
      process.env.BUN_SCIKIT_TREE_BACKEND = previousTreeBackend;
    }
  }
});
