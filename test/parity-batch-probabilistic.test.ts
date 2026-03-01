import { expect, test } from "bun:test";
import {
  EllipticEnvelope,
  GaussianProcessClassifier,
  GaussianProcessRegressor,
  GraphicalLasso,
  GraphicalLassoCV,
  IsotonicRegression,
} from "../src";

test("GaussianProcessRegressor and GaussianProcessClassifier fit/predict", () => {
  const XReg = [[0], [1], [2], [3], [4], [5]];
  const yReg = [0, 1, 2, 3, 4, 5];
  const gpr = new GaussianProcessRegressor({ alpha: 1e-6 }).fit(XReg, yReg);
  expect(gpr.predict(XReg).length).toBe(XReg.length);
  expect(gpr.predictStd(XReg).length).toBe(XReg.length);

  const XCls = [[-2], [-1], [0], [1], [2], [3]];
  const yCls = [0, 0, 0, 1, 1, 1];
  const gpc = new GaussianProcessClassifier({ alpha: 1e-6 }).fit(XCls, yCls);
  expect(gpc.predict(XCls).length).toBe(XCls.length);
  expect(gpc.predictProba(XCls).length).toBe(XCls.length);
});

test("GraphicalLasso family and EllipticEnvelope expose fitted covariance state", () => {
  const X = [
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 0],
    [1, 1, 0],
    [0.9, 1.1, -0.1],
    [0.1, -0.2, 1.1],
  ];

  const gl = new GraphicalLasso({ alpha: 0.05 }).fit(X);
  expect(gl.covariance_).not.toBeNull();
  expect(gl.scoreSamples(X).length).toBe(X.length);

  const glcv = new GraphicalLassoCV({ cv: 3, alphas: [0.01, 0.05, 0.1] }).fit(X);
  expect(glcv.alpha_).not.toBeNull();
  expect(glcv.scoreSamples(X).length).toBe(X.length);

  const envelope = new EllipticEnvelope({ contamination: 0.2 }).fit(X);
  expect(envelope.predict(X).length).toBe(X.length);
});

test("IsotonicRegression produces monotonic predictions", () => {
  const X = [[0], [1], [2], [3], [4], [5]];
  const y = [0, 1, 2, 2.1, 3.9, 5];
  const iso = new IsotonicRegression({ increasing: true }).fit(X, y);
  const pred = iso.predict(X);
  expect(pred.length).toBe(X.length);
  for (let i = 1; i < pred.length; i += 1) {
    expect(pred[i]).toBeGreaterThanOrEqual(pred[i - 1]);
  }
});

