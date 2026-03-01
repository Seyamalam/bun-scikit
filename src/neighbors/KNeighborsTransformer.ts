import type { Matrix } from "../types";
import { DistanceMetric, type DistanceMetricName } from "../metrics/DistanceMetric";
import { assertConsistentRowSize, assertFiniteMatrix, assertNonEmptyMatrix } from "../utils/validation";

export type KNeighborsTransformerMode = "distance" | "connectivity";

export interface KNeighborsTransformerOptions {
  nNeighbors?: number;
  mode?: KNeighborsTransformerMode;
  metric?: DistanceMetricName;
  p?: number;
}

export class KNeighborsTransformer {
  nFeaturesIn_: number | null = null;

  private nNeighbors: number;
  private mode: KNeighborsTransformerMode;
  private metricName: DistanceMetricName;
  private p: number;
  private metric: DistanceMetric;
  private XTrain: Matrix | null = null;
  private fitted = false;

  constructor(options: KNeighborsTransformerOptions = {}) {
    this.nNeighbors = options.nNeighbors ?? 5;
    this.mode = options.mode ?? "distance";
    this.metricName = options.metric ?? "minkowski";
    this.p = options.p ?? 2;
    this.metric = DistanceMetric.getMetric(this.metricName, { p: this.p });

    if (!Number.isInteger(this.nNeighbors) || this.nNeighbors < 1) {
      throw new Error(`nNeighbors must be an integer >= 1. Got ${this.nNeighbors}.`);
    }
    if (!(this.mode === "distance" || this.mode === "connectivity")) {
      throw new Error(`mode must be 'distance' or 'connectivity'. Got ${this.mode}.`);
    }
  }

  fit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    this.XTrain = X.map((row) => row.slice());
    this.nFeaturesIn_ = X[0].length;
    this.fitted = true;
    return this;
  }

  transform(X?: Matrix): Matrix {
    this.assertFitted();
    const query = X ?? this.XTrain!;
    assertNonEmptyMatrix(query);
    assertConsistentRowSize(query);
    assertFiniteMatrix(query);
    if (query[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${query[0].length}.`);
    }

    const out: Matrix = Array.from({ length: query.length }, () => new Array<number>(this.XTrain!.length).fill(0));
    const k = Math.min(this.nNeighbors, this.XTrain!.length);

    for (let i = 0; i < query.length; i += 1) {
      const neighbors = this.XTrain!
        .map((row, index) => ({ index, distance: this.metric.dist(query[i], row) }))
        .sort((a, b) => a.distance - b.distance)
        .slice(0, k);

      for (let j = 0; j < neighbors.length; j += 1) {
        const neighbor = neighbors[j];
        out[i][neighbor.index] = this.mode === "connectivity" ? 1 : neighbor.distance;
      }
    }
    return out;
  }

  fitTransform(X: Matrix): Matrix {
    return this.fit(X).transform(X);
  }

  private assertFitted(): void {
    if (!this.fitted || !this.XTrain || this.nFeaturesIn_ === null) {
      throw new Error("KNeighborsTransformer has not been fitted.");
    }
  }
}

