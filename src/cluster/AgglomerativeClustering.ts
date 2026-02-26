import type { Matrix, Vector } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";

export type AgglomerativeLinkage = "ward" | "complete" | "average" | "single";
export type AgglomerativeMetric = "euclidean";

export interface AgglomerativeClusteringOptions {
  nClusters?: number;
  linkage?: AgglomerativeLinkage;
  metric?: AgglomerativeMetric;
}

interface ClusterState {
  id: number;
  members: number[];
  centroid: number[];
}

function squaredEuclideanDistance(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

function euclideanDistance(a: number[], b: number[]): number {
  return Math.sqrt(squaredEuclideanDistance(a, b));
}

function centroidFromMembers(X: Matrix, members: number[]): number[] {
  const nFeatures = X[0].length;
  const center = new Array<number>(nFeatures).fill(0);
  for (let i = 0; i < members.length; i += 1) {
    const row = X[members[i]];
    for (let j = 0; j < nFeatures; j += 1) {
      center[j] += row[j];
    }
  }
  for (let j = 0; j < nFeatures; j += 1) {
    center[j] /= members.length;
  }
  return center;
}

export class AgglomerativeClustering {
  labels_: Vector | null = null;
  children_: number[][] | null = null;
  distances_: Vector | null = null;
  nConnectedComponents_ = 1;
  nLeaves_: number | null = null;
  nClusters_: number | null = null;
  nFeaturesIn_: number | null = null;

  private readonly nClusters: number;
  private readonly linkage: AgglomerativeLinkage;
  private readonly metric: AgglomerativeMetric;
  private isFitted = false;

  constructor(options: AgglomerativeClusteringOptions = {}) {
    this.nClusters = options.nClusters ?? 2;
    this.linkage = options.linkage ?? "ward";
    this.metric = options.metric ?? "euclidean";

    if (!Number.isInteger(this.nClusters) || this.nClusters < 1) {
      throw new Error(`nClusters must be an integer >= 1. Got ${this.nClusters}.`);
    }
    if (this.metric !== "euclidean") {
      throw new Error(`Only metric='euclidean' is currently supported. Got ${this.metric}.`);
    }
    if (
      this.linkage !== "ward" &&
      this.linkage !== "complete" &&
      this.linkage !== "average" &&
      this.linkage !== "single"
    ) {
      throw new Error(`Unsupported linkage '${this.linkage}'.`);
    }
  }

  fit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    const nSamples = X.length;
    if (this.nClusters > nSamples) {
      throw new Error(
        `nClusters (${this.nClusters}) cannot exceed sample count (${nSamples}).`,
      );
    }

    const children: number[][] = [];
    const distances: number[] = [];
    const active = new Map<number, ClusterState>();

    for (let i = 0; i < nSamples; i += 1) {
      active.set(i, {
        id: i,
        members: [i],
        centroid: X[i].slice(),
      });
    }

    let nextId = nSamples;
    while (active.size > 1) {
      let bestA = -1;
      let bestB = -1;
      let bestDistance = Number.POSITIVE_INFINITY;

      const ids = Array.from(active.keys()).sort((a, b) => a - b);
      for (let i = 0; i < ids.length; i += 1) {
        for (let j = i + 1; j < ids.length; j += 1) {
          const clusterA = active.get(ids[i])!;
          const clusterB = active.get(ids[j])!;
          const distance = this.clusterDistance(X, clusterA, clusterB);
          if (distance < bestDistance) {
            bestDistance = distance;
            bestA = ids[i];
            bestB = ids[j];
          }
        }
      }

      if (bestA < 0 || bestB < 0 || !Number.isFinite(bestDistance)) {
        throw new Error("AgglomerativeClustering failed to find a valid merge step.");
      }

      children.push([bestA, bestB]);
      distances.push(bestDistance);

      const mergedMembers = active.get(bestA)!.members.concat(active.get(bestB)!.members);
      const mergedState: ClusterState = {
        id: nextId,
        members: mergedMembers,
        centroid: centroidFromMembers(X, mergedMembers),
      };
      active.delete(bestA);
      active.delete(bestB);
      active.set(nextId, mergedState);
      nextId += 1;
    }

    const labels = this.labelsFromChildren(nSamples, children, this.nClusters);
    this.labels_ = labels;
    this.children_ = children;
    this.distances_ = distances;
    this.nLeaves_ = nSamples;
    this.nClusters_ = this.nClusters;
    this.nFeaturesIn_ = X[0].length;
    this.isFitted = true;
    return this;
  }

  fitPredict(X: Matrix): Vector {
    this.fit(X);
    return this.labels_!.slice();
  }

  private clusterDistance(X: Matrix, a: ClusterState, b: ClusterState): number {
    if (this.linkage === "ward") {
      const sizeA = a.members.length;
      const sizeB = b.members.length;
      const factor = (sizeA * sizeB) / (sizeA + sizeB);
      return Math.sqrt(factor * squaredEuclideanDistance(a.centroid, b.centroid));
    }

    if (this.linkage === "single") {
      let best = Number.POSITIVE_INFINITY;
      for (let i = 0; i < a.members.length; i += 1) {
        for (let j = 0; j < b.members.length; j += 1) {
          const distance = euclideanDistance(X[a.members[i]], X[b.members[j]]);
          if (distance < best) {
            best = distance;
          }
        }
      }
      return best;
    }

    if (this.linkage === "complete") {
      let best = Number.NEGATIVE_INFINITY;
      for (let i = 0; i < a.members.length; i += 1) {
        for (let j = 0; j < b.members.length; j += 1) {
          const distance = euclideanDistance(X[a.members[i]], X[b.members[j]]);
          if (distance > best) {
            best = distance;
          }
        }
      }
      return best;
    }

    let sum = 0;
    let count = 0;
    for (let i = 0; i < a.members.length; i += 1) {
      for (let j = 0; j < b.members.length; j += 1) {
        sum += euclideanDistance(X[a.members[i]], X[b.members[j]]);
        count += 1;
      }
    }
    return sum / count;
  }

  private labelsFromChildren(
    nSamples: number,
    children: number[][],
    targetClusters: number,
  ): number[] {
    const clusterMembers = new Map<number, number[]>();
    for (let i = 0; i < nSamples; i += 1) {
      clusterMembers.set(i, [i]);
    }

    const mergesToApply = nSamples - targetClusters;
    for (let mergeIndex = 0; mergeIndex < mergesToApply; mergeIndex += 1) {
      const [left, right] = children[mergeIndex];
      const nextId = nSamples + mergeIndex;
      const merged = clusterMembers.get(left)!.concat(clusterMembers.get(right)!);
      clusterMembers.delete(left);
      clusterMembers.delete(right);
      clusterMembers.set(nextId, merged);
    }

    const clusters = Array.from(clusterMembers.values()).sort(
      (a, b) => Math.min(...a) - Math.min(...b),
    );
    const labels = new Array<number>(nSamples).fill(0);
    for (let clusterIndex = 0; clusterIndex < clusters.length; clusterIndex += 1) {
      const members = clusters[clusterIndex];
      for (let i = 0; i < members.length; i += 1) {
        labels[members[i]] = clusterIndex;
      }
    }
    return labels;
  }

  private assertFitted(): void {
    if (!this.isFitted || !this.labels_ || !this.children_ || !this.distances_) {
      throw new Error("AgglomerativeClustering has not been fitted.");
    }
  }

  get nMerges_(): number {
    this.assertFitted();
    return this.children_!.length;
  }
}
