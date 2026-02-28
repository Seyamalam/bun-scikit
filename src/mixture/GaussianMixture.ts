import type { Matrix, Vector } from "../types";
import { argmax, normalizeProbabilitiesInPlace } from "../utils/classification";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";
import {
  covarianceRegularized,
  determinant,
  gaussianRandom,
  inverseRegularized,
  logSumExp,
  mulberry32,
} from "./shared";

export interface GaussianMixtureOptions {
  nComponents?: number;
  maxIter?: number;
  tolerance?: number;
  regCovar?: number;
  randomState?: number;
}

interface ComponentStats {
  inverse: Matrix;
  logDet: number;
}

export class GaussianMixture {
  weights_: Vector | null = null;
  means_: Matrix | null = null;
  covariances_: Matrix[] | null = null;
  converged_ = false;
  nIter_ = 0;
  lowerBound_ = Number.NEGATIVE_INFINITY;
  nFeaturesIn_: number | null = null;

  private nComponents: number;
  private maxIter: number;
  private tolerance: number;
  private regCovar: number;
  private randomState?: number;
  private fitted = false;

  constructor(options: GaussianMixtureOptions = {}) {
    this.nComponents = options.nComponents ?? 1;
    this.maxIter = options.maxIter ?? 100;
    this.tolerance = options.tolerance ?? 1e-3;
    this.regCovar = options.regCovar ?? 1e-6;
    this.randomState = options.randomState;
    this.validateOptions();
  }

  fit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    const nSamples = X.length;
    const nFeatures = X[0].length;
    if (this.nComponents > nSamples) {
      throw new Error(`nComponents (${this.nComponents}) cannot exceed sample count (${nSamples}).`);
    }

    const random = this.randomState === undefined ? Math.random : mulberry32(this.randomState);
    const indices = Array.from({ length: nSamples }, (_, idx) => idx);
    for (let i = indices.length - 1; i > 0; i -= 1) {
      const j = Math.floor(random() * (i + 1));
      const tmp = indices[i];
      indices[i] = indices[j];
      indices[j] = tmp;
    }

    this.weights_ = new Array<number>(this.nComponents).fill(1 / this.nComponents);
    this.means_ = new Array(this.nComponents);
    for (let k = 0; k < this.nComponents; k += 1) {
      this.means_[k] = X[indices[k]].slice();
    }
    this.covariances_ = new Array(this.nComponents);
    const globalCov = this.globalCovariance(X);
    for (let k = 0; k < this.nComponents; k += 1) {
      this.covariances_[k] = globalCov.map((row) => row.slice());
    }

    let prevLower = Number.NEGATIVE_INFINITY;
    this.converged_ = false;

    for (let iter = 0; iter < this.maxIter; iter += 1) {
      const stats = this.componentStats();
      const expectation = this.expectation(X, stats);
      const resp = expectation.responsibilities;
      this.lowerBound_ = expectation.lowerBound;

      this.maximization(X, resp);
      this.nIter_ = iter + 1;

      if (Math.abs(this.lowerBound_ - prevLower) < this.tolerance) {
        this.converged_ = true;
        break;
      }
      prevLower = this.lowerBound_;
    }

    this.nFeaturesIn_ = nFeatures;
    this.fitted = true;
    return this;
  }

  predictProba(X: Matrix): Matrix {
    this.assertFitted();
    this.validateInput(X);
    const expectation = this.expectation(X, this.componentStats());
    return expectation.responsibilities;
  }

  predict(X: Matrix): Vector {
    return this.predictProba(X).map((row) => argmax(row));
  }

  scoreSamples(X: Matrix): Vector {
    this.assertFitted();
    this.validateInput(X);
    const stats = this.componentStats();
    const out = new Array<number>(X.length);
    for (let i = 0; i < X.length; i += 1) {
      const scores = new Array<number>(this.nComponents);
      for (let k = 0; k < this.nComponents; k += 1) {
        scores[k] =
          Math.log(Math.max(1e-300, this.weights_![k])) +
          this.logGaussian(X[i], this.means_![k], stats[k]);
      }
      out[i] = logSumExp(scores);
    }
    return out;
  }

  score(X: Matrix): number {
    const values = this.scoreSamples(X);
    let sum = 0;
    for (let i = 0; i < values.length; i += 1) {
      sum += values[i];
    }
    return sum / values.length;
  }

  sample(nSamples = 1, randomState?: number): Matrix {
    this.assertFitted();
    if (!Number.isInteger(nSamples) || nSamples < 1) {
      throw new Error(`nSamples must be an integer >= 1. Got ${nSamples}.`);
    }
    const random = randomState === undefined ? Math.random : mulberry32(randomState);

    const cumulative = new Array<number>(this.weights_!.length).fill(0);
    let running = 0;
    for (let i = 0; i < this.weights_!.length; i += 1) {
      running += this.weights_![i];
      cumulative[i] = running;
    }

    const out: Matrix = new Array(nSamples);
    for (let i = 0; i < nSamples; i += 1) {
      const u = random();
      let component = 0;
      while (component < cumulative.length - 1 && u > cumulative[component]) {
        component += 1;
      }
      const mean = this.means_![component];
      const covariance = this.covariances_![component];
      const row = new Array<number>(mean.length);
      for (let j = 0; j < mean.length; j += 1) {
        const variance = Math.max(1e-12, covariance[j][j]);
        row[j] = mean[j] + gaussianRandom(random) * Math.sqrt(variance);
      }
      out[i] = row;
    }
    return out;
  }

  protected maximizeWeightsWithPrior(weights: Vector): Vector {
    return weights;
  }

  private expectation(
    X: Matrix,
    stats: ComponentStats[],
  ): { responsibilities: Matrix; lowerBound: number } {
    const responsibilities: Matrix = new Array(X.length);
    let lowerBound = 0;

    for (let i = 0; i < X.length; i += 1) {
      const scores = new Array<number>(this.nComponents);
      for (let k = 0; k < this.nComponents; k += 1) {
        scores[k] =
          Math.log(Math.max(1e-300, this.weights_![k])) +
          this.logGaussian(X[i], this.means_![k], stats[k]);
      }
      const logNorm = logSumExp(scores);
      lowerBound += logNorm;
      const row = new Array<number>(this.nComponents);
      for (let k = 0; k < this.nComponents; k += 1) {
        row[k] = Math.exp(scores[k] - logNorm);
      }
      normalizeProbabilitiesInPlace(row);
      responsibilities[i] = row;
    }

    return { responsibilities, lowerBound: lowerBound / X.length };
  }

  private maximization(X: Matrix, responsibilities: Matrix): void {
    const nSamples = X.length;
    const nFeatures = X[0].length;

    const nk = new Array<number>(this.nComponents).fill(0);
    for (let i = 0; i < nSamples; i += 1) {
      for (let k = 0; k < this.nComponents; k += 1) {
        nk[k] += responsibilities[i][k];
      }
    }

    const rawWeights = nk.map((value) => value / nSamples);
    this.weights_ = this.maximizeWeightsWithPrior(rawWeights);
    normalizeProbabilitiesInPlace(this.weights_);

    for (let k = 0; k < this.nComponents; k += 1) {
      const denom = Math.max(1e-12, nk[k]);
      const mean = new Array<number>(nFeatures).fill(0);
      for (let i = 0; i < nSamples; i += 1) {
        const r = responsibilities[i][k];
        for (let j = 0; j < nFeatures; j += 1) {
          mean[j] += r * X[i][j];
        }
      }
      for (let j = 0; j < nFeatures; j += 1) {
        mean[j] /= denom;
      }
      this.means_![k] = mean;

      const covariance: Matrix = Array.from({ length: nFeatures }, () =>
        new Array<number>(nFeatures).fill(0),
      );
      for (let i = 0; i < nSamples; i += 1) {
        const r = responsibilities[i][k];
        for (let a = 0; a < nFeatures; a += 1) {
          const da = X[i][a] - mean[a];
          for (let b = a; b < nFeatures; b += 1) {
            covariance[a][b] += r * da * (X[i][b] - mean[b]);
          }
        }
      }
      for (let a = 0; a < nFeatures; a += 1) {
        for (let b = a; b < nFeatures; b += 1) {
          covariance[a][b] /= denom;
          covariance[b][a] = covariance[a][b];
        }
      }
      this.covariances_![k] = covarianceRegularized(covariance, this.regCovar);
    }
  }

  private componentStats(): ComponentStats[] {
    const out: ComponentStats[] = new Array(this.nComponents);
    for (let k = 0; k < this.nComponents; k += 1) {
      const covariance = this.covariances_![k];
      out[k] = {
        inverse: inverseRegularized(covariance),
        logDet: Math.log(Math.max(1e-300, determinant(covariance))),
      };
    }
    return out;
  }

  private logGaussian(x: Vector, mean: Vector, stats: ComponentStats): number {
    const diff = new Array<number>(x.length);
    for (let i = 0; i < x.length; i += 1) {
      diff[i] = x[i] - mean[i];
    }
    const tmp = new Array<number>(x.length).fill(0);
    for (let i = 0; i < stats.inverse.length; i += 1) {
      for (let j = 0; j < stats.inverse[i].length; j += 1) {
        tmp[i] += stats.inverse[i][j] * diff[j];
      }
    }
    let quad = 0;
    for (let i = 0; i < diff.length; i += 1) {
      quad += diff[i] * tmp[i];
    }
    return -0.5 * (x.length * Math.log(2 * Math.PI) + stats.logDet + quad);
  }

  private globalCovariance(X: Matrix): Matrix {
    const mean = new Array<number>(X[0].length).fill(0);
    for (let i = 0; i < X.length; i += 1) {
      for (let j = 0; j < X[i].length; j += 1) {
        mean[j] += X[i][j];
      }
    }
    for (let j = 0; j < mean.length; j += 1) {
      mean[j] /= X.length;
    }

    const cov: Matrix = Array.from({ length: mean.length }, () =>
      new Array<number>(mean.length).fill(0),
    );
    for (let i = 0; i < X.length; i += 1) {
      for (let a = 0; a < mean.length; a += 1) {
        const da = X[i][a] - mean[a];
        for (let b = a; b < mean.length; b += 1) {
          cov[a][b] += da * (X[i][b] - mean[b]);
        }
      }
    }
    for (let a = 0; a < mean.length; a += 1) {
      for (let b = a; b < mean.length; b += 1) {
        cov[a][b] /= Math.max(1, X.length - 1);
        cov[b][a] = cov[a][b];
      }
    }
    return covarianceRegularized(cov, this.regCovar);
  }

  private validateInput(X: Matrix): void {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }
  }

  private validateOptions(): void {
    if (!Number.isInteger(this.nComponents) || this.nComponents < 1) {
      throw new Error(`nComponents must be an integer >= 1. Got ${this.nComponents}.`);
    }
    if (!Number.isInteger(this.maxIter) || this.maxIter < 1) {
      throw new Error(`maxIter must be an integer >= 1. Got ${this.maxIter}.`);
    }
    if (!Number.isFinite(this.tolerance) || this.tolerance < 0) {
      throw new Error(`tolerance must be finite and >= 0. Got ${this.tolerance}.`);
    }
    if (!Number.isFinite(this.regCovar) || this.regCovar < 0) {
      throw new Error(`regCovar must be finite and >= 0. Got ${this.regCovar}.`);
    }
  }

  private assertFitted(): void {
    if (!this.fitted || !this.weights_ || !this.means_ || !this.covariances_ || this.nFeaturesIn_ === null) {
      throw new Error("GaussianMixture has not been fitted.");
    }
  }
}
