export type Vector = number[];
export type Matrix = number[][];

export interface RegressionModel {
  fit(X: Matrix, y: Vector): this;
  predict(X: Matrix): Vector;
}

export interface ClassificationModel {
  fit(X: Matrix, y: Vector): this;
  predict(X: Matrix): Vector;
}
