import { PLSCanonical, type PLSCanonicalOptions } from "./PLSCanonical";

export interface CCAOptions extends PLSCanonicalOptions {
  copy?: boolean;
}

export class CCA extends PLSCanonical {
  copy: boolean;

  constructor(options: CCAOptions = {}) {
    super({
      ...options,
      scale: options.scale ?? true,
      maxIter: options.maxIter ?? 500,
      tolerance: options.tolerance ?? 1e-6,
    });
    this.copy = options.copy ?? true;
    this.validateOptions();
  }

  protected override validateOptions(): void {
    super.validateOptions();
    if (this.copy !== undefined && typeof this.copy !== "boolean") {
      throw new Error(`copy must be a boolean. Got ${this.copy as unknown as string}.`);
    }
  }
}
