import type { Matrix, Vector } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";
import { dot } from "../utils/linalg";

export interface IsomapOptions {
  nNeighbors?: number;
  nComponents?: number;
}

function euclideanDistance(a: Vector, b: Vector): number {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

function cloneMatrix(X: Matrix): Matrix {
  return X.map((row) => row.slice());
}

function pairwiseDistanceMatrix(X: Matrix): Matrix {
  const out: Matrix = Array.from({ length: X.length }, () => new Array<number>(X.length).fill(0));
  for (let i = 0; i < X.length; i += 1) {
    for (let j = i + 1; j < X.length; j += 1) {
      const dist = euclideanDistance(X[i], X[j]);
      out[i][j] = dist;
      out[j][i] = dist;
    }
  }
  return out;
}

function knnGraphDistances(dist: Matrix, nNeighbors: number): Matrix {
  const n = dist.length;
  const graph: Matrix = Array.from({ length: n }, () => new Array<number>(n).fill(Number.POSITIVE_INFINITY));
  for (let i = 0; i < n; i += 1) {
    graph[i][i] = 0;
    const order = Array.from({ length: n }, (_, idx) => idx)
      .filter((idx) => idx !== i)
      .sort((a, b) => dist[i][a] - dist[i][b])
      .slice(0, Math.min(nNeighbors, n - 1));
    for (let k = 0; k < order.length; k += 1) {
      const j = order[k];
      graph[i][j] = dist[i][j];
      graph[j][i] = dist[i][j];
    }
  }
  return graph;
}

function floydWarshall(graph: Matrix): Matrix {
  const n = graph.length;
  const out = graph.map((row) => row.slice());
  for (let k = 0; k < n; k += 1) {
    for (let i = 0; i < n; i += 1) {
      for (let j = 0; j < n; j += 1) {
        const via = out[i][k] + out[k][j];
        if (via < out[i][j]) {
          out[i][j] = via;
        }
      }
    }
  }
  return out;
}

function replaceInfiniteDistances(distances: Matrix): Matrix {
  let maxFinite = 0;
  for (let i = 0; i < distances.length; i += 1) {
    for (let j = 0; j < distances[i].length; j += 1) {
      const value = distances[i][j];
      if (Number.isFinite(value) && value > maxFinite) {
        maxFinite = value;
      }
    }
  }
  const penalty = Math.max(1, maxFinite * 10);
  const out = cloneMatrix(distances);
  for (let i = 0; i < out.length; i += 1) {
    for (let j = 0; j < out[i].length; j += 1) {
      if (!Number.isFinite(out[i][j])) {
        out[i][j] = penalty;
      }
    }
  }
  return out;
}

function jacobiEigenDecomposition(
  matrix: Matrix,
  tolerance = 1e-12,
  maxIter = 10_000,
): { eigenvalues: Vector; eigenvectors: Matrix } {
  const n = matrix.length;
  const A = cloneMatrix(matrix);
  const V: Matrix = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
  );

  for (let iter = 0; iter < maxIter; iter += 1) {
    let p = 0;
    let q = 1;
    let maxOffDiagonal = 0;
    for (let i = 0; i < n; i += 1) {
      for (let j = i + 1; j < n; j += 1) {
        const value = Math.abs(A[i][j]);
        if (value > maxOffDiagonal) {
          maxOffDiagonal = value;
          p = i;
          q = j;
        }
      }
    }

    if (maxOffDiagonal <= tolerance) {
      break;
    }

    const app = A[p][p];
    const aqq = A[q][q];
    const apq = A[p][q];
    const phi = 0.5 * Math.atan2(2 * apq, aqq - app);
    const c = Math.cos(phi);
    const s = Math.sin(phi);

    for (let i = 0; i < n; i += 1) {
      if (i === p || i === q) {
        continue;
      }
      const aip = A[i][p];
      const aiq = A[i][q];
      const newIP = c * aip - s * aiq;
      const newIQ = s * aip + c * aiq;
      A[i][p] = newIP;
      A[p][i] = newIP;
      A[i][q] = newIQ;
      A[q][i] = newIQ;
    }

    A[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
    A[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
    A[p][q] = 0;
    A[q][p] = 0;

    for (let i = 0; i < n; i += 1) {
      const vip = V[i][p];
      const viq = V[i][q];
      V[i][p] = c * vip - s * viq;
      V[i][q] = s * vip + c * viq;
    }
  }

  const eigenvalues = new Array<number>(n).fill(0);
  for (let i = 0; i < n; i += 1) {
    eigenvalues[i] = A[i][i];
  }
  return { eigenvalues, eigenvectors: V };
}

function classicalMds(distanceMatrix: Matrix, nComponents: number): {
  embedding: Matrix;
  eigenvectors: Matrix;
  eigenvalues: Vector;
  rowMeansSquared: Vector;
  grandMeanSquared: number;
} {
  const n = distanceMatrix.length;
  const squared: Matrix = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      squared[i][j] = distanceMatrix[i][j] * distanceMatrix[i][j];
    }
  }

  const rowMeansSquared = new Array<number>(n).fill(0);
  let grandMeanSquared = 0;
  for (let i = 0; i < n; i += 1) {
    let rowSum = 0;
    for (let j = 0; j < n; j += 1) {
      rowSum += squared[i][j];
    }
    rowMeansSquared[i] = rowSum / n;
    grandMeanSquared += rowMeansSquared[i];
  }
  grandMeanSquared /= n;

  const B: Matrix = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      B[i][j] = -0.5 * (squared[i][j] - rowMeansSquared[i] - rowMeansSquared[j] + grandMeanSquared);
    }
  }

  const { eigenvalues, eigenvectors } = jacobiEigenDecomposition(B);
  const order = Array.from({ length: n }, (_, index) => index).sort(
    (a, b) => eigenvalues[b] - eigenvalues[a],
  );
  const count = Math.min(nComponents, n);

  const embedding: Matrix = Array.from({ length: n }, () => new Array<number>(count).fill(0));
  const selectedEigenvalues = new Array<number>(count).fill(0);
  const selectedEigenvectors: Matrix = new Array(count);

  for (let c = 0; c < count; c += 1) {
    const eigenIndex = order[c];
    const eigenvalue = Math.max(0, eigenvalues[eigenIndex]);
    selectedEigenvalues[c] = eigenvalue;

    const vector = new Array<number>(n);
    for (let i = 0; i < n; i += 1) {
      vector[i] = eigenvectors[i][eigenIndex];
    }
    selectedEigenvectors[c] = vector;

    const scale = Math.sqrt(eigenvalue);
    for (let i = 0; i < n; i += 1) {
      embedding[i][c] = vector[i] * scale;
    }
  }

  return {
    embedding,
    eigenvectors: selectedEigenvectors,
    eigenvalues: selectedEigenvalues,
    rowMeansSquared,
    grandMeanSquared,
  };
}

export class Isomap {
  embedding_: Matrix | null = null;
  nFeaturesIn_: number | null = null;

  private nNeighbors: number;
  private nComponents: number;
  private XTrain: Matrix | null = null;
  private geodesicTrain: Matrix | null = null;
  private rowMeansSquared: Vector | null = null;
  private grandMeanSquared: number | null = null;
  private eigenvectors: Matrix | null = null;
  private eigenvalues: Vector | null = null;

  constructor(options: IsomapOptions = {}) {
    this.nNeighbors = options.nNeighbors ?? 5;
    this.nComponents = options.nComponents ?? 2;
    if (!Number.isInteger(this.nNeighbors) || this.nNeighbors < 1) {
      throw new Error(`nNeighbors must be an integer >= 1. Got ${this.nNeighbors}.`);
    }
    if (!Number.isInteger(this.nComponents) || this.nComponents < 1) {
      throw new Error(`nComponents must be an integer >= 1. Got ${this.nComponents}.`);
    }
  }

  fit(X: Matrix): this {
    this.embedding_ = this.fitTransform(X);
    return this;
  }

  fitTransform(X: Matrix): Matrix {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (this.nNeighbors >= X.length) {
      throw new Error(`nNeighbors must be < nSamples (${X.length}). Got ${this.nNeighbors}.`);
    }

    this.XTrain = X.map((row) => row.slice());
    this.nFeaturesIn_ = X[0].length;

    const dist = pairwiseDistanceMatrix(X);
    const graph = knnGraphDistances(dist, this.nNeighbors);
    const geodesic = replaceInfiniteDistances(floydWarshall(graph));
    this.geodesicTrain = geodesic;

    const mds = classicalMds(geodesic, this.nComponents);
    const embedding = mds.embedding;
    this.rowMeansSquared = mds.rowMeansSquared;
    this.grandMeanSquared = mds.grandMeanSquared;
    this.eigenvectors = mds.eigenvectors;
    this.eigenvalues = mds.eigenvalues;
    this.embedding_ = embedding.map((row) => row.slice());
    return embedding;
  }

  transform(X: Matrix): Matrix {
    if (
      !this.embedding_ ||
      !this.XTrain ||
      !this.geodesicTrain ||
      !this.rowMeansSquared ||
      this.grandMeanSquared === null ||
      !this.eigenvectors ||
      !this.eigenvalues ||
      this.nFeaturesIn_ === null
    ) {
      throw new Error("Isomap has not been fitted.");
    }
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }

    const k = Math.min(this.nNeighbors, this.XTrain.length);
    const trainCount = this.XTrain.length;
    const largeDistance = Math.max(1, this.grandMeanSquared);

    return X.map((row) => {
      const neighbors = this.XTrain!
        .map((trainRow, index) => ({ index, distance: euclideanDistance(row, trainRow) }))
        .sort((a, b) => a.distance - b.distance)
        .slice(0, k);

      const geodesicToTrain = new Array<number>(trainCount).fill(Number.POSITIVE_INFINITY);
      for (let i = 0; i < trainCount; i += 1) {
        let best = Number.POSITIVE_INFINITY;
        for (let j = 0; j < neighbors.length; j += 1) {
          const via = neighbors[j].distance + this.geodesicTrain![neighbors[j].index][i];
          if (via < best) {
            best = via;
          }
        }
        geodesicToTrain[i] = Number.isFinite(best) ? best : largeDistance;
      }

      const squaredDistances = geodesicToTrain.map((value) => value * value);
      let meanSquared = 0;
      for (let i = 0; i < squaredDistances.length; i += 1) {
        meanSquared += squaredDistances[i];
      }
      meanSquared /= squaredDistances.length;

      const centeredInner = new Array<number>(trainCount);
      for (let i = 0; i < trainCount; i += 1) {
        centeredInner[i] =
          -0.5 *
          (squaredDistances[i] - this.rowMeansSquared![i] - meanSquared + this.grandMeanSquared!);
      }

      const out = new Array<number>(this.eigenvectors!.length).fill(0);
      for (let c = 0; c < this.eigenvectors!.length; c += 1) {
        const denominator = Math.sqrt(Math.max(this.eigenvalues![c], 1e-12));
        if (denominator <= 1e-12) {
          out[c] = 0;
        } else {
          out[c] = dot(centeredInner, this.eigenvectors![c]) / denominator;
        }
      }
      return out;
    });
  }
}
