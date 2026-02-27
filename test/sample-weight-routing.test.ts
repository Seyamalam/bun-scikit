import { expect, test } from "bun:test";
import {
  ColumnTransformer,
  KFold,
  Pipeline,
  StackingClassifier,
  StackingRegressor,
  VotingClassifier,
  VotingRegressor,
} from "../src";

class RecordingTransformer {
  fitCalls: Array<number[] | undefined> = [];

  fit(_X: number[][], _y?: number[], sampleWeight?: number[]): this {
    this.fitCalls.push(sampleWeight ? sampleWeight.slice() : undefined);
    return this;
  }

  transform(X: number[][]): number[][] {
    return X.map((row) => row.slice());
  }
}

class RecordingClassifier {
  classes_ = [0, 1];
  fitCalls: Array<number[] | undefined> = [];

  fit(X: number[][], _y: number[], sampleWeight?: number[]): this {
    this.fitCalls.push(sampleWeight ? sampleWeight.slice() : undefined);
    this.classes_ = [0, 1];
    return this;
  }

  predict(X: number[][]): number[] {
    return X.map((row) => (row[0] >= 0 ? 1 : 0));
  }

  predictProba(X: number[][]): number[][] {
    return X.map((row) => (row[0] >= 0 ? [0.2, 0.8] : [0.8, 0.2]));
  }
}

class RecordingRegressor {
  fitCalls: Array<number[] | undefined> = [];

  fit(_X: number[][], _y: number[], sampleWeight?: number[]): this {
    this.fitCalls.push(sampleWeight ? sampleWeight.slice() : undefined);
    return this;
  }

  predict(X: number[][]): number[] {
    return X.map((row) => row[0]);
  }
}

test("Pipeline and ColumnTransformer route sampleWeight through fit", () => {
  const X = [[-2], [-1], [1], [2]];
  const y = [0, 0, 1, 1];
  const weights = [1, 2, 3, 4];

  const transformer = new RecordingTransformer();
  const finalEstimator = new RecordingClassifier();
  const pipeline = new Pipeline([
    ["t", transformer],
    ["clf", finalEstimator],
  ]);
  pipeline.fit(X, y, weights);
  expect(transformer.fitCalls[0]).toEqual(weights);
  expect(finalEstimator.fitCalls[0]).toEqual(weights);

  const noWeightTransformer = new RecordingTransformer();
  const noWeightFinalEstimator = new RecordingClassifier();
  const pipelineNoWeight = new Pipeline([
    ["t", noWeightTransformer],
    ["clf", noWeightFinalEstimator],
  ]).setFitRequest({ sampleWeight: false });
  pipelineNoWeight.fit(X, y, weights);
  expect(noWeightTransformer.fitCalls[0]).toBeUndefined();
  expect(noWeightFinalEstimator.fitCalls[0]).toBeUndefined();

  const ctTransformer = new RecordingTransformer();
  const columnTransformer = new ColumnTransformer([["keep", ctTransformer, [0]]]);
  columnTransformer.fit(X, y, weights);
  expect(ctTransformer.fitCalls[0]).toEqual(weights);

  const ctTransformerNoWeight = new RecordingTransformer();
  const columnTransformerNoWeight = new ColumnTransformer([["keep", ctTransformerNoWeight, [0]]])
    .setFitRequest({ sampleWeight: false });
  columnTransformerNoWeight.fit(X, y, weights);
  expect(ctTransformerNoWeight.fitCalls[0]).toBeUndefined();
});

test("Voting estimators propagate sampleWeight to base estimators", () => {
  const X = [[-2], [-1], [1], [2]];
  const yCls = [0, 0, 1, 1];
  const yReg = [1, 2, 3, 4];
  const weights = [1, 2, 3, 4];

  const clsA = new RecordingClassifier();
  const clsB = new RecordingClassifier();
  new VotingClassifier(
    [
      ["a", clsA],
      ["b", clsB],
    ],
    { voting: "soft" },
  ).fit(X, yCls, weights);
  expect(clsA.fitCalls[0]).toEqual(weights);
  expect(clsB.fitCalls[0]).toEqual(weights);

  const clsNoWeight = new RecordingClassifier();
  new VotingClassifier([["a", clsNoWeight]]).setFitRequest({ sampleWeight: false }).fit(X, yCls, weights);
  expect(clsNoWeight.fitCalls[0]).toBeUndefined();

  const regA = new RecordingRegressor();
  const regB = new RecordingRegressor();
  new VotingRegressor([
    ["a", regA],
    ["b", regB],
  ]).fit(X, yReg, weights);
  expect(regA.fitCalls[0]).toEqual(weights);
  expect(regB.fitCalls[0]).toEqual(weights);

  const regNoWeight = new RecordingRegressor();
  new VotingRegressor([["a", regNoWeight]]).setFitRequest({ sampleWeight: false }).fit(X, yReg, weights);
  expect(regNoWeight.fitCalls[0]).toBeUndefined();
});

test("Stacking estimators propagate fold and full sampleWeight", () => {
  const X = [[-3], [-2], [-1], [1], [2], [3]];
  const yCls = [0, 0, 0, 1, 1, 1];
  const yReg = [1, 2, 3, 4, 5, 6];
  const weights = [1, 2, 3, 4, 5, 6];

  const clsCreated: RecordingClassifier[] = [];
  const finalClsCreated: RecordingClassifier[] = [];
  const classifier = new StackingClassifier(
    [
      [
        "a",
        () => {
          const est = new RecordingClassifier();
          clsCreated.push(est);
          return est;
        },
      ],
    ],
    () => {
      const est = new RecordingClassifier();
      finalClsCreated.push(est);
      return est;
    },
    { cv: 3, randomState: 7 },
  );
  classifier.fit(X, yCls, weights);
  const clsFitWeights = clsCreated.map((est) => est.fitCalls[0]).filter((v) => v !== undefined) as number[][];
  expect(clsFitWeights.some((w) => w.length < weights.length)).toBe(true);
  expect(clsFitWeights.some((w) => w.length === weights.length)).toBe(true);
  expect(finalClsCreated[0].fitCalls[0]).toEqual(weights);

  const clsCreatedNoWeight: RecordingClassifier[] = [];
  new StackingClassifier(
    [["a", () => {
      const est = new RecordingClassifier();
      clsCreatedNoWeight.push(est);
      return est;
    }]],
    () => new RecordingClassifier(),
    { cv: 3, randomState: 7 },
  )
    .setFitRequest({ sampleWeight: false })
    .fit(X, yCls, weights);
  expect(clsCreatedNoWeight.every((est) => est.fitCalls[0] === undefined)).toBe(true);

  const regCreated: RecordingRegressor[] = [];
  const finalRegCreated: RecordingRegressor[] = [];
  const regressor = new StackingRegressor(
    [
      [
        "a",
        () => {
          const est = new RecordingRegressor();
          regCreated.push(est);
          return est;
        },
      ],
    ],
    () => {
      const est = new RecordingRegressor();
      finalRegCreated.push(est);
      return est;
    },
    { cv: 3, randomState: 7 },
  );
  regressor.fit(X, yReg, weights);
  const regFitWeights = regCreated.map((est) => est.fitCalls[0]).filter((v) => v !== undefined) as number[][];
  expect(regFitWeights.some((w) => w.length < weights.length)).toBe(true);
  expect(regFitWeights.some((w) => w.length === weights.length)).toBe(true);
  expect(finalRegCreated[0].fitCalls[0]).toEqual(weights);
});

test("StackingClassifier respects external CV splitters with sample weights", () => {
  const X = [[-3], [-2], [-1], [1], [2], [3]];
  const y = [0, 0, 0, 1, 1, 1];
  const weights = [1, 2, 3, 4, 5, 6];
  const baseCreated: RecordingClassifier[] = [];
  const finalCreated: RecordingClassifier[] = [];

  const model = new StackingClassifier(
    [
      [
        "a",
        () => {
          const est = new RecordingClassifier();
          baseCreated.push(est);
          return est;
        },
      ],
    ],
    () => {
      const est = new RecordingClassifier();
      finalCreated.push(est);
      return est;
    },
    { cv: 3, randomState: 11 },
  );
  model.fit(X, y, weights);
  expect(baseCreated.length).toBeGreaterThanOrEqual(new KFold({ nSplits: 3 }).split(X, y).length + 1);
  expect(finalCreated[0].fitCalls[0]).toEqual(weights);
});
