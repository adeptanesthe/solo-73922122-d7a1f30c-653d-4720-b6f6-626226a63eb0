"""Ridge regression with leave-one-factory-out lambda selection (NumPy).

Loss:  MSE + lambda * ||w||^2 , intercept unpenalised.

For every candidate lambda we rotate through the factories: hold one
out, standardise features with the *remaining* factories' pooled mean
and population standard deviation (zero-variance features are dropped),
fit on them, and score the held-out factory's MSE.  The lambda with the
smallest worst-case (max) held-out MSE wins; ties go to the smaller
lambda.  The chosen lambda is then refit on all factories.

Everything is computed from sufficient statistics only -- the service
never sees individual readings.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .stats import FEATURES, NAMES, TARGET, Stats, mean_and_cov, merge

LAMBDAS = (0.01, 0.1, 1.0)


@dataclass
class RidgeModel:
    """Ridge model on standardised features."""

    mean: list          # pooled training mean of (speed, pressure, power)
    std: dict           # feature index -> population std (kept features only)
    weights: list       # aligned with `kept`
    intercept: float    # unpenalised
    kept: list = field(default_factory=list)  # kept feature indices

    def predict(self, speed: float, pressure: float) -> float:
        x = {0: float(speed), 1: float(pressure)}
        y = self.intercept
        for w, i in zip(self.weights, self.kept):
            y += w * (x[i] - self.mean[i]) / self.std[i]
        return y

    def to_dict(self) -> dict:
        return {
            "mean": list(self.mean),
            "std": {str(k): v for k, v in self.std.items()},
            "weights": list(self.weights),
            "intercept": self.intercept,
            "kept": list(self.kept),
            "features": [NAMES[i] for i in self.kept],
        }

    @staticmethod
    def from_dict(d: dict) -> "RidgeModel":
        return RidgeModel(
            mean=list(d["mean"]),
            std={int(k): float(v) for k, v in d["std"].items()},
            weights=list(d["weights"]),
            intercept=float(d["intercept"]),
            kept=list(d["kept"]),
        )


def fit(mean, cov, lam: float) -> RidgeModel:
    """Closed-form ridge on standardised features.

    With population covariance C of the standardised features and
    c = Cov(z, y), minimising  (1/N)||y - b - Zw||^2 + lam*||w||^2
    gives  (C + lam*I) w = c  and  b = mean(y)  (intercept unpenalised).
    Zero-variance features are ignored.
    """
    kept = [i for i in FEATURES if cov[i][i] > 0.0]
    std = {i: math.sqrt(cov[i][i]) for i in kept}
    intercept = mean[TARGET]
    if not kept:
        return RidgeModel(mean=list(mean), std={}, weights=[],
                          intercept=intercept, kept=[])
    C = np.array(
        [[cov[i][j] / (std[i] * std[j]) for j in kept] for i in kept],
        dtype=np.float64,
    )
    c = np.array([cov[i][TARGET] / std[i] for i in kept], dtype=np.float64)
    A = C + lam * np.eye(len(kept))
    w = np.linalg.solve(A, c)
    return RidgeModel(mean=list(mean), std=std, weights=[float(v) for v in w],
                      intercept=intercept, kept=kept)


def mse_on(st: Stats, m: RidgeModel) -> float:
    """Exact MSE of a model on one factory, from its sufficient statistics.

    With r = intercept - mean(y):   MSE = Var(y) + r^2 + w'E[zz']w
        - 2 w'Cov(z, y) + 2 r w'E[z]
    which only uses exact covariances and means (numerically stable).
    """
    mean, cov = mean_and_cov(st)
    var_y = cov[TARGET][TARGET]
    r = m.intercept - mean[TARGET]
    mse = var_y + r * r
    if m.kept:
        ez = np.array([(mean[i] - m.mean[i]) / m.std[i] for i in m.kept])
        czz = np.array(
            [[cov[i][j] / (m.std[i] * m.std[j]) for j in m.kept] for i in m.kept]
        )
        ezz = czz + np.outer(ez, ez)
        czy = np.array([cov[i][TARGET] / m.std[i] for i in m.kept])
        w = np.array(m.weights)
        mse += float(w @ ezz @ w) - 2.0 * float(w @ czy) + 2.0 * r * float(w @ ez)
    return max(mse, 0.0)


@dataclass
class TrainingResult:
    lam: float
    factory_errors: list      # LOFO MSE per factory (same order as input)
    lambda_scores: dict       # lambda -> max held-out MSE
    model: RidgeModel


def train(factories) -> TrainingResult:
    """Full pipeline: LOFO lambda selection, then refit on all factories."""
    factories = list(factories)
    if not 3 <= len(factories) <= 8:
        raise ValueError("need between 3 and 8 factories")
    errors_by_lambda = {}
    scores = {}
    for lam in LAMBDAS:
        errs = []
        for k in range(len(factories)):
            train_stats = merge(f for j, f in enumerate(factories) if j != k)
            mean, cov = mean_and_cov(train_stats)
            model = fit(mean, cov, lam)
            errs.append(mse_on(factories[k], model))
        errors_by_lambda[lam] = errs
        scores[lam] = max(errs)
    # min() keeps the first (smallest) lambda on ties.
    best = min(LAMBDAS, key=lambda l: scores[l])
    mean, cov = mean_and_cov(merge(factories))
    model = fit(mean, cov, best)
    return TrainingResult(
        lam=best,
        factory_errors=errors_by_lambda[best],
        lambda_scores={lam: scores[lam] for lam in LAMBDAS},
        model=model,
    )
