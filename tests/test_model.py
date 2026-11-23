"""Ridge/LOFO pipeline: equivalence with a raw-row reference implementation."""
import math
import random

import numpy as np
import pytest

from app.model import LAMBDAS, fit, mse_on, train
from app.stats import Stats, apply_correction, mean_and_cov, merge


def make_factory_rows(n, speed_mu, press_mu, coef, intercept, noise, seed,
                      press_shift=0):
    """Simulate raw integer readings; pressure stored raw, true model uses
    pressure + press_shift."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        s = int(round(rng.gauss(speed_mu, 25)))
        p = int(round(rng.gauss(press_mu, 12)))
        y = coef[0] * s + coef[1] * (p + press_shift) + intercept + rng.gauss(0, noise)
        rows.append((s, p, int(round(y))))
    return rows


def stats_from_rows(rows):
    n = len(rows)
    sums = [sum(r[i] for r in rows) for i in range(3)]
    outer = [[sum(r[i] * r[j] for r in rows) for j in range(3)] for i in range(3)]
    return Stats.build(n, sums, outer)


# ---------------------------------------------------------------- reference
def reference_lofo(factory_rows, lam):
    """LOFO with raw rows: standardise by training mean/pop-std, ridge."""
    errs = []
    for k in range(len(factory_rows)):
        train_rows = [r for j, f in enumerate(factory_rows) if j != k for r in f]
        test_rows = factory_rows[k]
        X = np.array([[r[0], r[1]] for r in train_rows])
        y = np.array([r[2] for r in train_rows], dtype=float)
        mu = X.mean(axis=0)
        sd = X.std(axis=0)  # population std
        kept = [i for i in range(2) if sd[i] > 0]
        if kept:
            Z = (X[:, kept] - mu[kept]) / sd[kept]
            w = np.linalg.solve(Z.T @ Z + lam * len(Z) * np.eye(len(kept)),
                                Z.T @ (y - y.mean()))
            b = y.mean()
        else:
            w, b = np.zeros(0), y.mean()
        Xt = np.array([[r[0], r[1]] for r in test_rows])
        yt = np.array([r[2] for r in test_rows], dtype=float)
        if kept:
            pred = b + ((Xt[:, kept] - mu[kept]) / sd[kept]) @ w
        else:
            pred = np.full(len(yt), b)
        errs.append(float(np.mean((yt - pred) ** 2)))
    return errs


def test_pipeline_matches_raw_row_reference():
    factory_rows = [
        make_factory_rows(300, 1450, 120, (1.8, 0.6), 40, 5, seed=10 + i)
        for i in range(4)
    ]
    stats = [stats_from_rows(r) for r in factory_rows]
    result = train(stats)

    ref_scores = {lam: max(reference_lofo(factory_rows, lam)) for lam in LAMBDAS}
    ref_best = min(LAMBDAS, key=lambda l: ref_scores[l])
    assert result.lam == ref_best
    for lam in LAMBDAS:
        assert result.lambda_scores[lam] == pytest.approx(ref_scores[lam], rel=1e-9)

    ref_errs = reference_lofo(factory_rows, result.lam)
    for got, want in zip(result.factory_errors, ref_errs):
        assert got == pytest.approx(want, rel=1e-9)

    # final model == ridge on all raw rows with the chosen lambda
    all_rows = [r for f in factory_rows for r in f]
    X = np.array([[r[0], r[1]] for r in all_rows])
    y = np.array([r[2] for r in all_rows], dtype=float)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    Z = (X - mu) / sd
    w = np.linalg.solve(Z.T @ Z + result.lam * len(Z) * np.eye(2),
                        Z.T @ (y - y.mean()))
    assert result.model.intercept == pytest.approx(y.mean())
    assert result.model.weights == pytest.approx(list(w), rel=1e-9)
    # prediction matches the raw-row model
    for s, p, _ in all_rows[:20]:
        assert result.model.predict(s, p) == pytest.approx(
            y.mean() + float(((np.array([s, p]) - mu) / sd) @ w), rel=1e-9)


def test_correction_improves_miszeroed_factory():
    # F3's pressure sensor is off by +25 but reports correction 0.
    rows = [make_factory_rows(400, 1450, 120, (1.8, 0.6), 40, 5, seed=100 + i)
            for i in range(2)]
    rows.append(make_factory_rows(400, 1450, 120, (1.8, 0.6), 40, 5, seed=99,
                                  press_shift=25))
    rows.append(make_factory_rows(400, 1450, 120, (1.8, 0.6), 40, 5, seed=103))
    bad = [stats_from_rows(r) for r in rows]
    res_bad = train(bad)
    fixed = [apply_correction(s, 25) if i == 2 else s for i, s in enumerate(bad)]
    res_fixed = train(fixed)
    assert res_fixed.factory_errors[2] < res_bad.factory_errors[2] * 0.5


def test_zero_variance_feature_is_ignored():
    # speed constant everywhere -> only pressure used
    rng_rows = []
    for i in range(3):
        rows = [(1000, 100 + j % 7, 5 * (100 + j % 7) + 3) for j in range(50)]
        rng_rows.append(rows)
    stats = [stats_from_rows(r) for r in rng_rows]
    result = train(stats)
    assert result.model.kept == [1]
    # prediction must not depend on speed
    p1 = result.model.predict(1000, 105)
    p2 = result.model.predict(999999, 105)
    assert p1 == p2


def test_all_features_constant_predicts_mean():
    rows = [[(7, 7, 100 + i)] * 20 for i in range(3)]
    stats = [stats_from_rows(r) for r in rows]
    result = train(stats)
    assert result.model.kept == []
    assert result.model.predict(1, 2) == pytest.approx(101.0)


def test_lambda_tie_prefers_smaller():
    # constant features -> every lambda yields identical scores -> 0.01 wins
    rows = [[(5, 5, 10 * i + j % 3) for j in range(30)] for i in range(3)]
    stats = [stats_from_rows(r) for r in rows]
    result = train(stats)
    assert result.lam == 0.01
    assert len(set(result.lambda_scores.values())) == 1  # really tied


def test_huge_base_values_end_to_end():
    # all factories around 1e6 counts; variance must survive standardisation
    factory_rows = [
        make_factory_rows(200, 1_000_000 + 50 * i, 500_000 + 10 * i,
                          (0.02, 0.01), 1000, 3, seed=1000 + i)
        for i in range(4)
    ]
    stats = [stats_from_rows(r) for r in factory_rows]
    result = train(stats)
    assert result.model.kept == [0, 1]
    assert result.model.std[0] > 0 and result.model.std[1] > 0
    # LOFO errors should be tiny compared to the power scale (~1e5)
    assert max(result.factory_errors) < 1e4
