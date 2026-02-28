import { expect, test } from "bun:test";
import {
  BayesianGaussianMixture,
  GaussianMixture,
} from "../src";

const X = [
  [0.0, 0.1],
  [0.1, -0.1],
  [-0.1, 0.0],
  [2.0, 2.1],
  [2.2, 1.9],
  [1.9, 2.0],
];

test("GaussianMixture fits two clusters and predicts responsibilities", () => {
  const gmm = new GaussianMixture({
    nComponents: 2,
    maxIter: 50,
    randomState: 42,
  }).fit(X);

  expect(gmm.weights_).not.toBeNull();
  expect(gmm.means_).not.toBeNull();
  const proba = gmm.predictProba([[0, 0], [2.1, 2.0]]);
  expect(proba.length).toBe(2);
  expect(proba[0].length).toBe(2);
  const pred = gmm.predict([[0, 0], [2.1, 2.0]]);
  expect(pred[0]).not.toBe(pred[1]);
});

test("BayesianGaussianMixture fits and can sample", () => {
  const bgmm = new BayesianGaussianMixture({
    nComponents: 3,
    maxIter: 50,
    randomState: 7,
    weightConcentrationPrior: 1.5,
  }).fit(X);

  expect(bgmm.weights_).not.toBeNull();
  const samples = bgmm.sample(4, 12);
  expect(samples.length).toBe(4);
  expect(samples[0].length).toBe(2);
});
