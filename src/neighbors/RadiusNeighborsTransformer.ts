import type { Matrix } from "../types";
import { DistanceMetric, type DistanceMetricName } from "../metrics/DistanceMetric";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";

export type RadiusNeighborsTransformerMode = "distance" | "connectivity";

export interface RadiusNeighborsTransformerOptions {
  radius?: number;
  mode?: RadiusNeighborsTransformerMode;
  metric?: DistanceMetricName;
  p?: number;
}

export class RadiusNeighborsTransformer {
  nFeaturesIn_: number | null = null;

  private readonly radius: number;
  private readonly mode: RadiusNeighborsTransformerMode;
  private readonly metricName: DistanceMetricName;
  private readonly p: number;
  private readonly metric: DistanceMetric;
  private XTrain: Matrix | null = null;
  private fitted = false;

  constructor(options: RadiusNeighborsTransformerOptions = {}) {
    this.radius = options.radius ?? 1;
    this.mode = options.mode ?? "distance";
    this.metricName = options.metric ?? "minkowski";
    this.p = options.p ?? 2;
    this.metric = DistanceMetric.getMetric(this.metricName, { p: this.p });

    if (!Number.isFinite(this.radius) || this.radius <= 0) {
      throw new Error(`radius must be finite and > 0. Got ${this.radius}.`);
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
    for (let i = 0; i < query.length; i += 1) {
      for (let j = 0; j < this.XTrain!.length; j += 1) {
        const distance = this.metric.dist(query[i], this.XTrain![j]);
        if (distance > this.radius) {
          continue;
        }
        out[i][j] = this.mode === "connectivity" ? 1 : distance;
      }
    }
    return out;
  }

  fitTransform(X: Matrix): Matrix {
    return this.fit(X).transform(X);
  }

  private assertFitted(): void {
    if (!this.fitted || !this.XTrain || this.nFeaturesIn_ === null) {
      throw new Error("RadiusNeighborsTransformer has not been fitted.");
    }
  }
}
