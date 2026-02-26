import type { Matrix, Vector } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";

export interface DBSCANOptions {
  eps?: number;
  minSamples?: number;
}

function squaredEuclideanDistance(a: Vector, b: Vector): number {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

export class DBSCAN {
  labels_: Vector | null = null;
  coreSampleIndices_: number[] | null = null;
  components_: Matrix | null = null;
  nFeaturesIn_: number | null = null;

  private readonly eps: number;
  private readonly minSamples: number;
  private isFitted = false;

  constructor(options: DBSCANOptions = {}) {
    this.eps = options.eps ?? 0.5;
    this.minSamples = options.minSamples ?? 5;

    if (!Number.isFinite(this.eps) || this.eps <= 0) {
      throw new Error(`eps must be finite and > 0. Got ${this.eps}.`);
    }
    if (!Number.isInteger(this.minSamples) || this.minSamples < 1) {
      throw new Error(`minSamples must be an integer >= 1. Got ${this.minSamples}.`);
    }
  }

  fit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    const nSamples = X.length;
    const labels = new Array<number>(nSamples).fill(-99);
    const visited = new Array<boolean>(nSamples).fill(false);
    const epsSquared = this.eps * this.eps;
    const coreIndices: number[] = [];
    const neighborhoodCache = new Map<number, number[]>();
    let clusterId = 0;

    const regionQuery = (sampleIndex: number): number[] => {
      const cached = neighborhoodCache.get(sampleIndex);
      if (cached) {
        return cached;
      }

      const neighbors: number[] = [];
      const source = X[sampleIndex];
      for (let candidate = 0; candidate < nSamples; candidate += 1) {
        if (squaredEuclideanDistance(source, X[candidate]) <= epsSquared) {
          neighbors.push(candidate);
        }
      }
      neighborhoodCache.set(sampleIndex, neighbors);
      return neighbors;
    };

    for (let sampleIndex = 0; sampleIndex < nSamples; sampleIndex += 1) {
      if (visited[sampleIndex]) {
        continue;
      }

      visited[sampleIndex] = true;
      const neighbors = regionQuery(sampleIndex);

      if (neighbors.length < this.minSamples) {
        labels[sampleIndex] = -1;
        continue;
      }

      const seenInQueue = new Set<number>(neighbors);
      const queue = neighbors.slice();
      labels[sampleIndex] = clusterId;

      while (queue.length > 0) {
        const neighborIndex = queue.shift()!;
        if (!visited[neighborIndex]) {
          visited[neighborIndex] = true;
          const neighborNeighbors = regionQuery(neighborIndex);
          if (neighborNeighbors.length >= this.minSamples) {
            for (let i = 0; i < neighborNeighbors.length; i += 1) {
              const candidate = neighborNeighbors[i];
              if (!seenInQueue.has(candidate)) {
                queue.push(candidate);
                seenInQueue.add(candidate);
              }
            }
          }
        }

        if (labels[neighborIndex] === -1 || labels[neighborIndex] === -99) {
          labels[neighborIndex] = clusterId;
        }
      }

      clusterId += 1;
    }

    for (let sampleIndex = 0; sampleIndex < nSamples; sampleIndex += 1) {
      if (labels[sampleIndex] === -99) {
        labels[sampleIndex] = -1;
      }
    }

    for (let sampleIndex = 0; sampleIndex < nSamples; sampleIndex += 1) {
      const neighbors = regionQuery(sampleIndex);
      if (neighbors.length >= this.minSamples) {
        coreIndices.push(sampleIndex);
      }
    }

    this.labels_ = labels;
    this.coreSampleIndices_ = coreIndices;
    this.components_ = coreIndices.map((index) => X[index].slice());
    this.nFeaturesIn_ = X[0].length;
    this.isFitted = true;
    return this;
  }

  fitPredict(X: Matrix): Vector {
    this.fit(X);
    return this.labels_!.slice();
  }

  private assertFitted(): void {
    if (!this.isFitted || !this.labels_ || !this.coreSampleIndices_ || !this.components_) {
      throw new Error("DBSCAN has not been fitted.");
    }
  }

  get nClusters_(): number {
    this.assertFitted();
    const unique = new Set<number>();
    for (let i = 0; i < this.labels_!.length; i += 1) {
      if (this.labels_![i] >= 0) {
        unique.add(this.labels_![i]);
      }
    }
    return unique.size;
  }
}
