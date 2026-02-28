import type { Vector } from "../types";
import { GaussianMixture, type GaussianMixtureOptions } from "./GaussianMixture";

export interface BayesianGaussianMixtureOptions extends GaussianMixtureOptions {
  weightConcentrationPrior?: number;
}

export class BayesianGaussianMixture extends GaussianMixture {
  private weightConcentrationPrior: number;

  constructor(options: BayesianGaussianMixtureOptions = {}) {
    super(options);
    this.weightConcentrationPrior = options.weightConcentrationPrior ?? 1;
    if (!Number.isFinite(this.weightConcentrationPrior) || this.weightConcentrationPrior <= 0) {
      throw new Error(
        `weightConcentrationPrior must be finite and > 0. Got ${this.weightConcentrationPrior}.`,
      );
    }
  }

  protected override maximizeWeightsWithPrior(weights: Vector): Vector {
    const k = weights.length;
    const out = new Array<number>(k);
    const pseudoCount = this.weightConcentrationPrior / k;
    let sum = 0;
    for (let i = 0; i < k; i += 1) {
      out[i] = weights[i] + pseudoCount;
      sum += out[i];
    }
    for (let i = 0; i < k; i += 1) {
      out[i] /= Math.max(1e-12, sum);
    }
    return out;
  }
}
