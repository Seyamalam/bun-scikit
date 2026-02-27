export type Vector = number[];
export type Matrix = number[][];

export interface Transformer {
  fit(X: Matrix, y?: Vector, sampleWeight?: Vector): this;
  transform(X: Matrix): Matrix;
  fitTransform?(X: Matrix, y?: Vector, sampleWeight?: Vector): Matrix;
}

export interface RegressionModel {
  fit(X: Matrix, y: Vector, sampleWeight?: Vector): this;
  predict(X: Matrix): Vector;
}

export interface ClassificationModel {
  fit(X: Matrix, y: Vector, sampleWeight?: Vector): this;
  predict(X: Matrix): Vector;
}
