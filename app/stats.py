"""Exact sufficient-statistics arithmetic.

Factories only report *integer* vector sums and outer-product sums for
readings x = (speed, pressure, power).  All aggregation is done with
Python integers, so no precision is ever lost -- even when the raw
readings have huge base values (e.g. speed ~ 1e9 with a variance of a
few units).  The covariance numerator

    C[i][j] = N * sum(x_i * x_j) - sum(x_i) * sum(x_j)

is computed exactly as an integer and only converted to float at the
very end, which keeps non-zero variance intact where the naive
float64 identity  E[x^2] - E[x]^2  would cancel catastrophically.
"""
from __future__ import annotations

from dataclasses import dataclass

DIM = 3
FEATURES = (0, 1)  # speed, pressure
TARGET = 2         # power
NAMES = ("speed", "pressure", "power")


class InvalidStats(ValueError):
    """Raised when reported sums cannot come from real readings."""


@dataclass(frozen=True)
class Stats:
    """Sufficient statistics for one factory (or a pooled set)."""

    n: int
    sums: tuple  # (sum speed, sum pressure, sum power) -- ints
    outer: tuple  # 3x3 tuple of tuples, sum of x x^T -- ints

    @staticmethod
    def build(n: int, sums, outer) -> "Stats":
        sums = tuple(int(v) for v in sums)
        outer = tuple(tuple(int(v) for v in row) for row in outer)
        st = Stats(n=int(n), sums=sums, outer=outer)
        validate(st)
        return st


def validate(st: Stats) -> None:
    """Exact consistency checks (all integer arithmetic)."""
    if st.n < 1:
        raise InvalidStats("sample count n must be >= 1")
    if len(st.sums) != DIM:
        raise InvalidStats("sums must have length 3")
    if len(st.outer) != DIM or any(len(r) != DIM for r in st.outer):
        raise InvalidStats("outer must be 3x3")
    o = st.outer
    for i in range(DIM):
        if o[i][i] < 0:
            raise InvalidStats(f"outer[{i}][{i}] (sum of squares) is negative")
        for j in range(DIM):
            if o[i][j] != o[j][i]:
                raise InvalidStats("outer-product sums must be symmetric")
    # The covariance numerator must be positive semidefinite.  For a 3x3
    # matrix PSD <=> all principal minors >= 0; computed exactly in ints.
    c = cov_numerator(st)
    for i in range(DIM):
        if c[i][i] < 0:
            raise InvalidStats(
                f"inconsistent sums: n*sum({NAMES[i]}^2) < sum({NAMES[i]})^2"
            )
    for i in range(DIM):
        for j in range(i + 1, DIM):
            if c[i][i] * c[j][j] - c[i][j] * c[i][j] < 0:
                raise InvalidStats(
                    f"inconsistent sums: covariance minor ({i},{j}) is negative"
                )
    det = (
        c[0][0] * (c[1][1] * c[2][2] - c[1][2] * c[1][2])
        - c[0][1] * (c[0][1] * c[2][2] - c[1][2] * c[0][2])
        + c[0][2] * (c[0][1] * c[1][2] - c[1][1] * c[0][2])
    )
    if det < 0:
        raise InvalidStats("inconsistent sums: covariance determinant is negative")


def apply_correction(st: Stats, correction: int) -> Stats:
    """Shift pressure by an integer zero-point correction, exactly.

    p' = p + c  implies
        sum(p')    = sum(p) + n*c
        sum(p'^2)  = sum(p^2) + 2c*sum(p) + n*c^2
        sum(s p')  = sum(s p) + c*sum(s)
        sum(p' y)  = sum(p y) + c*sum(y)
    """
    c = int(correction)
    if c == 0:
        return st
    n = st.n
    s = list(st.sums)
    o = [list(row) for row in st.outer]
    o[1][1] = o[1][1] + 2 * c * s[1] + n * c * c
    o[0][1] = o[0][1] + c * s[0]
    o[1][0] = o[0][1]
    o[1][2] = o[1][2] + c * s[2]
    o[2][1] = o[1][2]
    s[1] = s[1] + n * c
    return Stats(n=n, sums=tuple(s), outer=tuple(tuple(r) for r in o))


def merge(many) -> Stats:
    """Pool sufficient statistics of several factories (exact)."""
    many = list(many)
    if not many:
        raise InvalidStats("no factories to pool")
    n = sum(s.n for s in many)
    sums = tuple(sum(s.sums[i] for s in many) for i in range(DIM))
    outer = tuple(
        tuple(sum(s.outer[i][j] for s in many) for j in range(DIM))
        for i in range(DIM)
    )
    return Stats(n=n, sums=sums, outer=outer)


def cov_numerator(st: Stats):
    """Exact integer matrix  C = N*sum(x x^T) - sum(x) sum(x)^T."""
    n = st.n
    return [
        [n * st.outer[i][j] - st.sums[i] * st.sums[j] for j in range(DIM)]
        for i in range(DIM)
    ]


def mean_and_cov(st: Stats):
    """Population mean and covariance as floats.

    The covariance is exact-integer until the final division, so a
    non-zero variance survives however large the reading base value is.
    """
    n = st.n
    mean = [s / n for s in st.sums]
    c = cov_numerator(st)
    denom = n * n
    cov = [[c[i][j] / denom for j in range(DIM)] for i in range(DIM)]
    return mean, cov
