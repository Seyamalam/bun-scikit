import type { ClassificationModel, Matrix, Vector } from "../types";
import { accuracyScore } from "../metrics/classification";
import { DecisionTreeClassifier, type MaxFeaturesOption } from "../tree/DecisionTreeClassifier";
import { assertFiniteVector, validateClassificationInputs } from "../utils/validation";

export interface RandomForestClassifierOptions {
  nEstimators?: number;
  maxDepth?: number;
  minSamplesSplit?: number;
  minSamplesLeaf?: number;
  maxFeatures?: MaxFeaturesOption;
  bootstrap?: boolean;
  randomState?: number;
}

function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class RandomForestClassifier implements ClassificationModel {
  classes_: Vector = [0, 1];
  private readonly nEstimators: number;
  private readonly maxDepth?: number;
  private readonly minSamplesSplit?: number;
  private readonly minSamplesLeaf?: number;
  private readonly maxFeatures: MaxFeaturesOption;
  private readonly bootstrap: boolean;
  private readonly randomState?: number;
  private trees: DecisionTreeClassifier[] = [];

  constructor(options: RandomForestClassifierOptions = {}) {
    this.nEstimators = options.nEstimators ?? 50;
    this.maxDepth = options.maxDepth ?? 12;
    this.minSamplesSplit = options.minSamplesSplit ?? 2;
    this.minSamplesLeaf = options.minSamplesLeaf ?? 1;
    this.maxFeatures = options.maxFeatures ?? "sqrt";
    this.bootstrap = options.bootstrap ?? true;
    this.randomState = options.randomState;

    if (!Number.isInteger(this.nEstimators) || this.nEstimators < 1) {
      throw new Error(`nEstimators must be a positive integer. Got ${this.nEstimators}.`);
    }
  }

  fit(X: Matrix, y: Vector): this {
    validateClassificationInputs(X, y);

    const sampleCount = X.length;
    const random = this.randomState === undefined ? Math.random : mulberry32(this.randomState);
    this.trees = [];

    for (let estimatorIndex = 0; estimatorIndex < this.nEstimators; estimatorIndex += 1) {
      const sampleIndices = new Array<number>(sampleCount);
      if (this.bootstrap) {
        for (let i = 0; i < sampleCount; i += 1) {
          sampleIndices[i] = Math.floor(random() * sampleCount);
        }
      } else {
        for (let i = 0; i < sampleCount; i += 1) {
          sampleIndices[i] = i;
        }
      }

      const tree = new DecisionTreeClassifier({
        maxDepth: this.maxDepth,
        minSamplesSplit: this.minSamplesSplit,
        minSamplesLeaf: this.minSamplesLeaf,
        maxFeatures: this.maxFeatures,
        randomState:
          this.randomState === undefined ? undefined : this.randomState + estimatorIndex + 1,
      });
      tree.fit(X, y, sampleIndices, true);
      this.trees.push(tree);
    }

    return this;
  }

  predict(X: Matrix): Vector {
    if (this.trees.length === 0) {
      throw new Error("RandomForestClassifier has not been fitted.");
    }

    const treePredictions = this.trees.map((tree) => tree.predict(X));
    const sampleCount = X.length;
    const predictions = new Array(sampleCount).fill(0);

    for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex += 1) {
      let positiveVotes = 0;
      for (let treeIndex = 0; treeIndex < treePredictions.length; treeIndex += 1) {
        positiveVotes += treePredictions[treeIndex][sampleIndex] === 1 ? 1 : 0;
      }
      predictions[sampleIndex] = positiveVotes * 2 >= this.trees.length ? 1 : 0;
    }

    return predictions;
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return accuracyScore(y, this.predict(X));
  }
}
