import { PLSRegression, type PLSRegressionOptions } from "./PLSRegression";

export interface PLSCanonicalOptions extends PLSRegressionOptions {}

export class PLSCanonical extends PLSRegression {
  constructor(options: PLSCanonicalOptions = {}) {
    super(options);
    this.deflationMode = "canonical";
  }
}
