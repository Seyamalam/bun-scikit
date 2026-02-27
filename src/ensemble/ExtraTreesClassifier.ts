import type { ClassificationModel, Matrix, Vector } from "../types";
import { accuracyScore } from "../metrics/classification";
import { assertFiniteVector } from "../utils/validation";
import {
  RandomForestClassifier,
  type RandomForestClassifierOptions,
} from "./RandomForestClassifier";

export interface ExtraTreesClassifierOptions extends RandomForestClassifierOptions {}

export class ExtraTreesClassifier implements ClassificationModel {
  classes_: Vector = [0, 1];
  featureImportances_: Vector | null = null;
  fitBackend_: "zig" | "js" = "js";
  fitBackendLibrary_: string | null = null;

  private options: ExtraTreesClassifierOptions;
  private forest: RandomForestClassifier;

  constructor(options: ExtraTreesClassifierOptions = {}) {
    this.options = {
      nEstimators: options.nEstimators ?? 100,
      maxDepth: options.maxDepth ?? undefined,
      minSamplesSplit: options.minSamplesSplit ?? 2,
      minSamplesLeaf: options.minSamplesLeaf ?? 1,
      maxFeatures: options.maxFeatures ?? "sqrt",
      bootstrap: options.bootstrap ?? false,
      randomState: options.randomState,
    };
    this.forest = new RandomForestClassifier(this.options);
  }

  getParams(): ExtraTreesClassifierOptions {
    return { ...this.options };
  }

  setParams(params: Partial<ExtraTreesClassifierOptions>): this {
    this.options = { ...this.options, ...params };
    this.forest = new RandomForestClassifier(this.options);
    return this;
  }

  fit(X: Matrix, y: Vector): this {
    this.forest.fit(X, y);
    this.classes_ = this.forest.classes_.slice();
    this.featureImportances_ = this.forest.featureImportances_?.slice() ?? null;
    this.fitBackend_ = this.forest.fitBackend_;
    this.fitBackendLibrary_ = this.forest.fitBackendLibrary_;
    return this;
  }

  predict(X: Matrix): Vector {
    return this.forest.predict(X);
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return accuracyScore(y, this.predict(X));
  }

  dispose(): void {
    this.forest.dispose();
  }
}
