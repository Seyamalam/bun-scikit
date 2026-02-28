import { expect, test } from "bun:test";
import { CCA, PLSCanonical, PLSRegression, PLSSVD } from "../src";

const X = [
  [0.5, 1.2, -0.3],
  [1.0, 0.9, -0.1],
  [1.4, 0.6, 0.2],
  [2.1, -0.2, 0.9],
  [2.5, -0.7, 1.1],
  [3.0, -1.0, 1.4],
];

const yVector = [0.2, 0.5, 0.9, 1.7, 2.0, 2.4];
const YMatrix = yVector.map((value, i) => [value, value * 0.4 + X[i][0] * 0.2]);

test("PLSSVD fit/transform exposes latent projections", () => {
  const model = new PLSSVD({ nComponents: 2 }).fit(X, YMatrix);
  const embedding = model.transform(X);
  expect(embedding.length).toBe(X.length);
  expect(embedding[0].length).toBe(2);
  expect(model.xWeights_).not.toBeNull();
  expect(model.yWeights_).not.toBeNull();
  expect(model.singularValues_?.length).toBe(2);
});

test("PLSRegression predicts vector and matrix targets", () => {
  const vectorModel = new PLSRegression({ nComponents: 2 }).fit(X, yVector);
  const vectorPred = vectorModel.predict(X);
  expect(Array.isArray(vectorPred)).toBe(true);
  expect((vectorPred as number[]).length).toBe(X.length);

  const matrixModel = new PLSRegression({ nComponents: 2 }).fit(X, YMatrix);
  const matrixPred = matrixModel.predict(X);
  expect((matrixPred as number[][]).length).toBe(X.length);
  expect((matrixPred as number[][])[0].length).toBe(2);
  expect(matrixModel.coef_?.length).toBe(2);
});

test("PLSCanonical and CCA return paired score spaces", () => {
  const canonical = new PLSCanonical({ nComponents: 2 }).fit(X, YMatrix);
  const canonicalScores = canonical.transform(X, YMatrix);
  expect(canonicalScores[0].length).toBe(X.length);
  expect(canonicalScores[0][0].length).toBe(2);
  expect(canonicalScores[1].length).toBe(X.length);
  expect(canonicalScores[1][0].length).toBe(2);

  const cca = new CCA({ nComponents: 2 }).fit(X, YMatrix);
  const ccaScores = cca.transform(X, YMatrix);
  expect(ccaScores[0].length).toBe(X.length);
  expect(ccaScores[1].length).toBe(X.length);
});
