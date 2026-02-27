import { expect, test } from "bun:test";
import { LinearSVR } from "../src/svm/LinearSVR";
import { NuSVC } from "../src/svm/NuSVC";
import { NuSVR } from "../src/svm/NuSVR";
import { SVC } from "../src/svm/SVC";
import { SVR } from "../src/svm/SVR";

test("SVC fits a separable binary classification problem", () => {
  const X = [[-3], [-2], [-1], [1], [2], [3]];
  const y = [0, 0, 0, 1, 1, 1];
  const model = new SVC({ kernel: "linear", C: 1, maxIter: 200, learningRate: 0.1 }).fit(X, y);
  expect(model.score(X, y)).toBeGreaterThan(0.95);
  expect(model.predict([[-2.5], [2.5]])).toEqual([0, 1]);
});

test("SVR learns a smooth regression mapping", () => {
  const X = [[0], [1], [2], [3], [4], [5]];
  const y = [1, 3, 5, 7, 9, 11];
  const model = new SVR({ kernel: "linear", C: 5 }).fit(X, y);
  const pred = model.predict([[6]])[0];
  expect(pred).toBeGreaterThan(10);
  expect(model.score(X, y)).toBeGreaterThan(0.95);
});

test("LinearSVR fits a linear target", () => {
  const X = [[0], [1], [2], [3], [4], [5]];
  const y = [1, 3, 5, 7, 9, 11];
  const model = new LinearSVR({ C: 5, epsilon: 0.01, maxIter: 12000, learningRate: 0.03 }).fit(X, y);
  expect(model.score(X, y)).toBeGreaterThan(0.9);
});

test("NuSVC/ NuSVR expose sklearn-like nu interfaces", () => {
  const clsX = [[-2], [-1], [1], [2]];
  const clsY = [0, 0, 1, 1];
  const nuSvc = new NuSVC({ nu: 0.4, kernel: "linear", maxIter: 200, learningRate: 0.1 }).fit(clsX, clsY);
  expect(nuSvc.predict([[-1.5], [1.5]])).toEqual([0, 1]);

  const regX = [[0], [1], [2], [3]];
  const regY = [0, 2, 4, 6];
  const nuSvr = new NuSVR({ nu: 0.4, kernel: "linear", C: 3 }).fit(regX, regY);
  expect(nuSvr.predict([[4]])[0]).toBeGreaterThan(6.5);
});
