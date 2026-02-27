import type { Matrix, RegressionModel, Vector } from "../types";
import { r2Score } from "../metrics/regression";
import { assertFiniteVector } from "../utils/validation";
import { SVR, type SVROptions } from "./SVR";
import type { GammaOption, KernelName } from "./kernelUtils";

export interface NuSVROptions {
  nu?: number;
  C?: number;
  kernel?: KernelName;
  gamma?: GammaOption;
  degree?: number;
  coef0?: number;
}

export class NuSVR implements RegressionModel {
  dualCoef_: Vector = [];
  intercept_ = 0;
  support_: number[] = [];
  supportVectors_: Matrix = [];

  private nu: number;
  private readonly svr: SVR;

  constructor(options: NuSVROptions = {}) {
    this.nu = options.nu ?? 0.5;
    if (!Number.isFinite(this.nu) || this.nu <= 0 || this.nu >= 1) {
      throw new Error(`nu must be in (0, 1). Got ${this.nu}.`);
    }
    this.svr = new SVR(this.toSVROptions(options));
  }

  getParams(): NuSVROptions {
    const params = this.svr.getParams();
    return {
      nu: this.nu,
      C: params.C,
      kernel: params.kernel,
      gamma: params.gamma,
      degree: params.degree,
      coef0: params.coef0,
    };
  }

  setParams(params: Partial<NuSVROptions>): this {
    if (params.nu !== undefined) {
      this.nu = params.nu;
      if (!Number.isFinite(this.nu) || this.nu <= 0 || this.nu >= 1) {
        throw new Error(`nu must be in (0, 1). Got ${this.nu}.`);
      }
    }
    this.svr.setParams(this.toSVROptions({ ...this.getParams(), ...params }));
    return this;
  }

  fit(X: Matrix, y: Vector): this {
    this.svr.fit(X, y);
    this.dualCoef_ = this.svr.dualCoef_.slice();
    this.intercept_ = this.svr.intercept_;
    this.support_ = this.svr.support_.slice();
    this.supportVectors_ = this.svr.supportVectors_.map((row) => row.slice());
    return this;
  }

  predict(X: Matrix): Vector {
    return this.svr.predict(X);
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return r2Score(y, this.predict(X));
  }

  private toSVROptions(options: NuSVROptions): SVROptions {
    const nu = options.nu ?? this.nu;
    const epsilon = Math.max(1e-6, 0.5 * (1 - nu));
    return {
      C: options.C ?? 1.0,
      epsilon,
      kernel: options.kernel,
      gamma: options.gamma,
      degree: options.degree,
      coef0: options.coef0,
    };
  }
}
