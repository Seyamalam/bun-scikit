import { expect, test } from "bun:test";
import { trainTestSplit } from "../src/model_selection/trainTestSplit";

test("trainTestSplit creates deterministic split with randomState", () => {
  const X = [0, 1, 2, 3, 4, 5];
  const y = X.map((v) => v * 10);

  const splitA = trainTestSplit(X, y, { testSize: 2, randomState: 123 });
  const splitB = trainTestSplit(X, y, { testSize: 2, randomState: 123 });

  expect(splitA).toEqual(splitB);
  expect(splitA.XTest.length).toBe(2);
  expect(splitA.XTrain.length).toBe(4);
  expect(splitA.yTest.length).toBe(2);
  expect(splitA.yTrain.length).toBe(4);
});
