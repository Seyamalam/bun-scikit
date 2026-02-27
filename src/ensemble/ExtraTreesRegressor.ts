import type { Matrix, RegressionModel, Vector } from "../types";
import { r2Score } from "../metrics/regression";
import { assertFiniteVector } from "../utils/validation";
import {
  RandomForestRegressor,
  type RandomForestRegressorOptions,
} from "./RandomForestRegressor";

export interface ExtraTreesRegressorOptions extends RandomForestRegressorOptions {}

export class ExtraTreesRegressor implements RegressionModel {
  featureImportances_: Vector | null = null;

  private options: ExtraTreesRegressorOptions;
  private forest: RandomForestRegressor;

  constructor(options: ExtraTreesRegressorOptions = {}) {
    this.options = {
      nEstimators: options.nEstimators ?? 100,
      maxDepth: options.maxDepth ?? undefined,
      minSamplesSplit: options.minSamplesSplit ?? 2,
      minSamplesLeaf: options.minSamplesLeaf ?? 1,
      maxFeatures: options.maxFeatures ?? 1.0,
      bootstrap: options.bootstrap ?? false,
      randomState: options.randomState,
    };
    this.forest = new RandomForestRegressor(this.options);
  }

  getParams(): ExtraTreesRegressorOptions {
    return { ...this.options };
  }

  setParams(params: Partial<ExtraTreesRegressorOptions>): this {
    this.options = { ...this.options, ...params };
    this.forest = new RandomForestRegressor(this.options);
    return this;
  }

  fit(X: Matrix, y: Vector): this {
    this.forest.fit(X, y);
    this.featureImportances_ = this.forest.featureImportances_?.slice() ?? null;
    return this;
  }

  predict(X: Matrix): Vector {
    return this.forest.predict(X);
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return r2Score(y, this.predict(X));
  }
}
