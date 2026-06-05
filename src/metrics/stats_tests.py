"""Statistical tests: proportions, chi-square, McNemar, paired."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats


@dataclass
class TestResult:
    hypothesis_id: str
    test_id: str
    description: str
    level: str  # row | family
    family_name: str | None
    n1: int | None = None
    k1: int | None = None
    n2: int | None = None
    k2: int | None = None
    p1: float | None = None
    p2: float | None = None
    effect: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    statistic: float | None = None
    p_raw: float | None = None
    q_value: float | None = None
    rejected_fdr: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "test_id": self.test_id,
            "description": self.description,
            "level": self.level,
            "family_name": self.family_name,
            "n1": self.n1,
            "k1": self.k1,
            "n2": self.n2,
            "k2": self.k2,
            "p1": self.p1,
            "p2": self.p2,
            "effect": self.effect,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "statistic": self.statistic,
            "p_raw": self.p_raw,
            "q_value": self.q_value,
            "rejected_fdr": self.rejected_fdr,
            **{f"extra_{k}": v for k, v in self.extra.items()},
        }


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n) / denom
    return (center - margin, center + margin)


def proportion_summary(k: int, n: int) -> dict[str, float]:
    p = k / n if n else float("nan")
    lo, hi = _wilson_ci(k, n)
    return {"k": k, "n": n, "p": p, "ci_low": lo, "ci_high": hi}


def one_proportion_ztest(k: int, n: int, p0: float = 0.5, alternative: str = "two-sided") -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    phat = k / n
    # Degenerate null (p0=0 or 1): normal SE is 0 — use exact binomial.
    if p0 <= 0.0:
        alt = "two-sided" if alternative == "two-sided" else "greater"
        p = float(stats.binomtest(k, n, p0, alternative=alt).pvalue)
        return (np.inf if phat > p0 else 0.0, p)
    if p0 >= 1.0:
        alt = "two-sided" if alternative == "two-sided" else "less"
        p = float(stats.binomtest(k, n, p0, alternative=alt).pvalue)
        return (-np.inf if phat < p0 else 0.0, p)
    se = math.sqrt(p0 * (1 - p0) / n)
    z = (phat - p0) / se
    if alternative == "two-sided":
        p = 2 * stats.norm.sf(abs(z))
    elif alternative == "larger":
        p = stats.norm.sf(z)
    else:
        p = stats.norm.cdf(z)
    return (z, p)


def two_proportion_ztest(
    k1: int, n1: int, k2: int, n2: int, alternative: str = "two-sided"
) -> tuple[float, float, float, tuple[float, float]]:
    if n1 == 0 or n2 == 0:
        return (np.nan, np.nan, np.nan, (np.nan, np.nan))
    p1, p2 = k1 / n1, k2 / n2
    pooled = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        if alternative == "larger":
            p = 1.0 if p1 <= p2 else 0.0
            z = 0.0 if p1 == p2 else (np.inf if p1 > p2 else -np.inf)
            return (z, p, p1 - p2, (np.nan, np.nan))
        if alternative == "smaller":
            p = 1.0 if p1 >= p2 else 0.0
            z = 0.0 if p1 == p2 else (-np.inf if p1 < p2 else np.inf)
            return (z, p, p1 - p2, (np.nan, np.nan))
        p = 1.0 if p1 == p2 else 0.0
        z = 0.0 if p1 == p2 else np.inf
        return (z, p, p1 - p2, (np.nan, np.nan))
    z = (p1 - p2) / se
    if alternative == "two-sided":
        p = 2 * stats.norm.sf(abs(z))
    elif alternative == "larger":
        p = stats.norm.sf(z)
    else:
        p = stats.norm.cdf(z)
    se_diff = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    ci = (p1 - p2 - 1.96 * se_diff, p1 - p2 + 1.96 * se_diff)
    return (z, p, p1 - p2, ci)


def chi2_independence(table: np.ndarray) -> tuple[float, float, int, np.ndarray]:
    chi2, p, dof, expected = stats.chi2_contingency(table)
    return (float(chi2), float(p), int(dof), expected)


def mcnemar_from_discordant(b: int, c: int, *, exact: bool = False) -> tuple[float, float]:
    """b: man=Yes,woman=No; c: man=No,woman=Yes."""
    if b + c == 0:
        return (0.0, 1.0)
    if exact and b + c < 25:
        p = float(stats.binomtest(min(b, c), b + c, 0.5).pvalue) * 2
        p = min(p, 1.0)
        return (np.nan, p)
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p = float(stats.chi2.sf(stat, 1))
    return (stat, p)


def mcnemar_greater_discordant(b: int, c: int) -> tuple[float, float]:
    """One-sided McNemar: b > c (e.g. Yes with abstain vs Yes without). b+c discordant pairs."""
    n = b + c
    if n == 0:
        return (0.0, 1.0)
    if n < 25:
        p = float(stats.binomtest(b, n, 0.5, alternative="greater").pvalue)
        return (np.nan, p)
    if b <= c:
        stat = 0.0
        p = 0.5 if b == c else float(stats.binomtest(b, n, 0.5, alternative="greater").pvalue)
    else:
        stat = (b - c - 1) ** 2 / n
        p = float(stats.chi2.sf(stat, 1))
    return (stat, p)


def mcnemar_smaller_discordant(b: int, c: int) -> tuple[float, float]:
    """One-sided McNemar: c > b (e.g. Yes without abstain more than with)."""
    return mcnemar_greater_discordant(c, b)


def one_sample_ttest_vs(x: np.ndarray, mu: float = 0.0) -> tuple[float, float]:
    """One-sample t-test vs mu; exact limits when all values are identical."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    if len(x) == 1:
        if x[0] == mu:
            return (0.0, 1.0)
        return (np.inf if x[0] > mu else -np.inf, 0.0)
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1))
    if std == 0.0 or np.isclose(std, 0.0):
        if np.isclose(mean, mu):
            return (0.0, 1.0)
        return (np.inf if mean > mu else -np.inf, 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        t, p = stats.ttest_1samp(x, mu)
    return (float(t), float(p))


def wilcoxon_vs(x: np.ndarray, mu: float = 0.0) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    try:
        w, p = stats.wilcoxon(x - mu, alternative="two-sided")
        return (float(w), float(p))
    except ValueError:
        return (0.0, 1.0)


def pearson_r(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return (np.nan, np.nan)
    if np.std(x) == 0 or np.std(y) == 0:
        return (np.nan, np.nan)
    r, p = stats.pearsonr(x, y)
    return (float(r), float(p))
