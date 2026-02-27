import type { ClassificationModel, Matrix, Vector } from "../types";
import { accuracyScore } from "../metrics/classification";
import { assertFiniteVector } from "../utils/validation";
import { SVC, type SVCOptions } from "./SVC";
import type { GammaOption, KernelName } from "./kernelUtils";

export interface NuSVCOptions {
  nu?: number;
  kernel?: KernelName;
  gamma?: GammaOption;
  degree?: number;
  coef0?: number;
  maxIter?: number;
  tolerance?: number;
  learningRate?: number;
}

export class NuSVC implements ClassificationModel {
  classes_: Vector = [];
  supportVectors_: Matrix = [];
  support_: number[] = [];

  private nu: number;
  private readonly svc: SVC;

  constructor(options: NuSVCOptions = {}) {
    this.nu = options.nu ?? 0.5;
    if (!Number.isFinite(this.nu) || this.nu <= 0 || this.nu >= 1) {
      throw new Error(`nu must be in (0, 1). Got ${this.nu}.`);
    }
    this.svc = new SVC(this.toSVCOptions(options));
  }

  getParams(): NuSVCOptions {
    const svcParams = this.svc.getParams();
    return {
      nu: this.nu,
      kernel: svcParams.kernel,
      gamma: svcParams.gamma,
      degree: svcParams.degree,
      coef0: svcParams.coef0,
      maxIter: svcParams.maxIter,
      tolerance: svcParams.tolerance,
      learningRate: svcParams.learningRate,
    };
  }

  setParams(params: Partial<NuSVCOptions>): this {
    if (params.nu !== undefined) {
      this.nu = params.nu;
      if (!Number.isFinite(this.nu) || this.nu <= 0 || this.nu >= 1) {
        throw new Error(`nu must be in (0, 1). Got ${this.nu}.`);
      }
    }
    this.svc.setParams(this.toSVCOptions({ ...this.getParams(), ...params }));
    return this;
  }

  fit(X: Matrix, y: Vector, sampleWeight?: Vector): this {
    this.svc.fit(X, y);
    this.classes_ = this.svc.classes_.slice();
    this.support_ = this.svc.support_.slice();
    this.supportVectors_ = this.svc.supportVectors_.map((row) => row.slice());
    return this;
  }

  decisionFunction(X: Matrix): Vector | Matrix {
    return this.svc.decisionFunction(X);
  }

  predictProba(X: Matrix): Matrix {
    return this.svc.predictProba(X);
  }

  predict(X: Matrix): Vector {
    return this.svc.predict(X);
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return accuracyScore(y, this.predict(X));
  }

  private toSVCOptions(options: NuSVCOptions): SVCOptions {
    const nu = options.nu ?? this.nu;
    const c = Math.max(1e-6, (1 / (1 - nu)) * 0.5);
    return {
      C: c,
      kernel: options.kernel,
      gamma: options.gamma,
      degree: options.degree,
      coef0: options.coef0,
      maxIter: options.maxIter,
      tolerance: options.tolerance,
      learningRate: options.learningRate,
    };
  }
}

