import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import {
  CalibratedClassifierCV,
  GaussianNB,
  KNeighborsClassifier,
  NMF,
  VotingClassifier,
} from "../src";

function meanAbsDiff(a: number[][], b: number[][]): number {
  let total = 0;
  let count = 0;
  for (let i = 0; i < a.length; i += 1) {
    for (let j = 0; j < a[i].length; j += 1) {
      total += Math.abs(a[i][j] - b[i][j]);
      count += 1;
    }
  }
  return total / count;
}

function meanSquaredError(a: number[][], b: number[][]): number {
  let total = 0;
  let count = 0;
  for (let i = 0; i < a.length; i += 1) {
    for (let j = 0; j < a[i].length; j += 1) {
      const diff = a[i][j] - b[i][j];
      total += diff * diff;
      count += 1;
    }
  }
  return total / count;
}

const fixture = JSON.parse(
  readFileSync(resolve("test/fixtures/sklearn-snapshots.json"), "utf-8"),
);

test("GaussianNB probabilities stay close to sklearn snapshot", () => {
  const model = new GaussianNB().fit(fixture.multiclass.X, fixture.multiclass.y);
  const proba = model.predictProba(fixture.multiclass.probe);
  expect(meanAbsDiff(proba, fixture.multiclass.gaussian_nb_proba)).toBeLessThan(0.12);
});

test("VotingClassifier soft probabilities stay close to sklearn snapshot", () => {
  const model = new VotingClassifier(
    [
      ["gnb", () => new GaussianNB()],
      ["knn", () => new KNeighborsClassifier({ nNeighbors: 3 })],
    ],
    { voting: "soft" },
  ).fit(fixture.multiclass.X, fixture.multiclass.y);

  const proba = model.predictProba(fixture.multiclass.probe);
  expect(meanAbsDiff(proba, fixture.multiclass.voting_soft_proba)).toBeLessThan(0.15);
});

test("CalibratedClassifierCV probabilities stay close to sklearn snapshot", () => {
  const model = new CalibratedClassifierCV(() => new GaussianNB(), {
    cv: 3,
    method: "sigmoid",
    ensemble: false,
    randomState: 42,
  }).fit(fixture.multiclass.X, fixture.multiclass.y);

  const proba = model.predictProba(fixture.multiclass.probe);
  expect(meanAbsDiff(proba, fixture.multiclass.calibrated_sigmoid_proba)).toBeLessThan(0.2);
});

test("NMF reconstruction stays close to sklearn snapshot", () => {
  const model = new NMF({
    nComponents: 2,
    maxIter: 500,
    tolerance: 1e-6,
    randomState: 42,
  });
  const W = model.fitTransform(fixture.nmf.X);
  const reconstruction = model.inverseTransform(W);
  expect(meanSquaredError(reconstruction, fixture.nmf.reconstruction)).toBeLessThan(0.05);
});
