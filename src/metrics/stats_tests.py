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


SHAPIRO_MAX_N = 5000


def shapiro_wilk_normality(
    x: np.ndarray,
    *,
    alpha: float = 0.05,
    max_n: int = SHAPIRO_MAX_N,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Shapiro–Wilk test for normality (diagnostic for prob t-tests).

    scipy.stats.shapiro accepts at most max_n (5000) samples; larger arrays are
    deterministically subsampled.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return {
            "shapiro_stat": float("nan"),
            "shapiro_p": float("nan"),
            "shapiro_n": n,
            "shapiro_n_used": n,
            "shapiro_subsampled": False,
            "shapiro_normal": None,
            "shapiro_alpha": alpha,
            "shapiro_note": "n<3: Shapiro–Wilk not applicable",
        }
    subsampled = n > max_n
    if subsampled:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_n, replace=False)
        sample = np.sort(x[idx])
        note = f"subsampled n={max_n} of {n} (Shapiro–Wilk limit)"
    else:
        sample = x
        note = f"full sample n={n}"
    stat, p = stats.shapiro(sample)
    p = float(p)
    return {
        "shapiro_stat": float(stat),
        "shapiro_p": p,
        "shapiro_n": n,
        "shapiro_n_used": len(sample),
        "shapiro_subsampled": subsampled,
        "shapiro_normal": bool(p >= alpha),
        "shapiro_alpha": alpha,
        "shapiro_note": note,
    }


def gender_preference_ztest(
    man: int,
    woman: int,
    alternative: str = "two-sided",
) -> tuple[float, float, float, tuple[float, float]]:
    """
    H₀: P(man | man or woman) = 0.5 — conditional binomial among non-abstain answers.

    z = (M - W) / sqrt(M + W). Not a two-proportion z-test on the full sample.
    """
    n = man + woman
    if n == 0:
        return (float("nan"), float("nan"), float("nan"), (float("nan"), float("nan")))

    theta_hat = man / n
    z = (man - woman) / math.sqrt(n)

    if alternative == "two-sided":
        p = 2 * stats.norm.sf(abs(z))
    elif alternative == "greater":
        p = stats.norm.sf(z)
    elif alternative == "less":
        p = stats.norm.cdf(z)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    se_theta = math.sqrt(theta_hat * (1 - theta_hat) / n)
    ci_theta = (theta_hat - 1.96 * se_theta, theta_hat + 1.96 * se_theta)
    return (float(z), float(p), float(theta_hat), ci_theta)


def gender_preference_test(
    man: int,
    woman: int,
    alternative: str = "two-sided",
) -> dict[str, Any]:
    """Conditional binomial test: M among N=M+W; normal z + exact binomial."""
    n = man + woman
    if n == 0:
        return {
            "man": man,
            "woman": woman,
            "non_abstain": 0,
            "theta_hat": float("nan"),
            "conditional_gap": float("nan"),
            "z": float("nan"),
            "p_norm": float("nan"),
            "p_exact": float("nan"),
            "ci_theta_wald_95": (float("nan"), float("nan")),
        }

    z, p_norm, theta_hat, ci_theta = gender_preference_ztest(
        man, woman, alternative=alternative
    )
    gap = (man - woman) / n
    exact = stats.binomtest(k=man, n=n, p=0.5, alternative=alternative)

    return {
        "man": man,
        "woman": woman,
        "non_abstain": n,
        "theta_hat": theta_hat,
        "conditional_gap": gap,
        "z": z,
        "p_norm": p_norm,
        "p_exact": float(exact.pvalue),
        "ci_theta_wald_95": ci_theta,
    }


def trinomial_choice_analysis(
    man: int,
    woman: int,
    abstain: int,
    alternative: str = "two-sided",
) -> dict[str, Any]:
    """
    Trinomial sample: gender preference among M+W (conditional binomial) +
    separate abstain rate A/(M+W+A).
    """
    total = man + woman + abstain
    non_abstain = man + woman

    if total == 0:
        return {
            "total": 0,
            "man": man,
            "woman": woman,
            "abstain": abstain,
            "preference_among_non_abstain": None,
            "p_man_raw": float("nan"),
            "p_woman_raw": float("nan"),
            "p_abstain": float("nan"),
            "answer_rate": float("nan"),
            "abstain_rate": float("nan"),
            "raw_gap": float("nan"),
        }

    preference = (
        gender_preference_test(man, woman, alternative=alternative)
        if non_abstain > 0
        else None
    )

    abstain_rate = abstain / total
    answer_rate = non_abstain / total
    se_abstain = math.sqrt(abstain_rate * (1 - abstain_rate) / total)
    ci_abstain = (
        abstain_rate - 1.96 * se_abstain,
        abstain_rate + 1.96 * se_abstain,
    )

    return {
        "total": total,
        "man": man,
        "woman": woman,
        "abstain": abstain,
        "p_man_raw": man / total,
        "p_woman_raw": woman / total,
        "p_abstain": abstain_rate,
        "answer_rate": answer_rate,
        "abstain_rate": abstain_rate,
        "ci_abstain_wald_95": ci_abstain,
        "raw_gap": (man - woman) / total,
        "preference_among_non_abstain": preference,
    }


def trinomial_man_vs_woman(
    k_man: int,
    k_woman: int,
    k_abstain: int | None = None,
    *,
    n: int | None = None,
) -> dict[str, Any]:
    """
    Test H₀: P(man) = P(woman) under a trinomial (man / woman / abstain) model.

    - 3 categories: Pearson χ² and G² (likelihood-ratio) with 1 df.
      Abstain proportion is estimated freely; only p_man = p_woman is tested.
    - 2 categories (k_abstain=0): equivalent to binomial H₀ p_man=0.5 among gender
      choices; also reports exact binomial two-sided p.
    """
    if n is None:
        if k_abstain is None:
            n = k_man + k_woman
            k_abstain = 0
        else:
            n = k_man + k_woman + k_abstain
    if k_abstain is None:
        k_abstain = max(0, n - k_man - k_woman)

    if n <= 0:
        return {
            "n": n,
            "k_man": k_man,
            "k_woman": k_woman,
            "k_abstain": k_abstain,
            "n_categories": 0,
            "effect_p_man_minus_p_woman": float("nan"),
            "pearson_chi2": float("nan"),
            "pearson_p": float("nan"),
            "g2": float("nan"),
            "g2_p": float("nan"),
            "binom_exact_p": float("nan"),
            "method": "invalid",
        }

    p_man = k_man / n
    p_woman = k_woman / n
    effect = p_man - p_woman
    gender_total = k_man + k_woman

    if gender_total == 0:
        return {
            "n": n,
            "k_man": k_man,
            "k_woman": k_woman,
            "k_abstain": k_abstain,
            "n_categories": 3 if k_abstain > 0 else 2,
            "effect_p_man_minus_p_woman": 0.0,
            "pearson_chi2": 0.0,
            "pearson_p": 1.0,
            "g2": 0.0,
            "g2_p": 1.0,
            "binom_exact_p": 1.0,
            "method": "degenerate_no_gender_choice",
        }

    # Pearson χ²: E[man]=E[woman]=(k_man+k_woman)/2 under H₀ p_man=p_woman
    expected_gender = gender_total / 2.0
    pearson_chi2 = (
        (k_man - expected_gender) ** 2 + (k_woman - expected_gender) ** 2
    ) / expected_gender
    pearson_p = float(stats.chi2.sf(pearson_chi2, 1))

    # G² (likelihood-ratio) for same H₀
    def _g(x: float, total: int) -> float:
        if x <= 0:
            return 0.0
        p = x / total
        return x * math.log(p)

    ll_full = _g(k_man, n) + _g(k_woman, n) + _g(k_abstain, n)
    p_g_null = gender_total / (2.0 * n)
    p_a_null = k_abstain / n
    ll_null = gender_total * math.log(p_g_null) + k_abstain * math.log(p_a_null) if p_a_null > 0 else gender_total * math.log(p_g_null)
    if k_abstain == 0:
        ll_null = gender_total * math.log(0.5)
    g2 = max(0.0, 2.0 * (ll_full - ll_null))
    g2_p = float(stats.chi2.sf(g2, 1))

    binom_exact_p = float("nan")
    if k_abstain == 0:
        binom_exact_p = float(
            stats.binomtest(k_man, gender_total, 0.5, alternative="two-sided").pvalue
        )

    n_cat = 3 if k_abstain > 0 else 2
    method = (
        "trinomial_pearson_g2_1df"
        if n_cat == 3
        else "binomial_pearson_g2_exact_1df"
    )

    return {
        "n": n,
        "k_man": k_man,
        "k_woman": k_woman,
        "k_abstain": k_abstain,
        "n_categories": n_cat,
        "p_man": p_man,
        "p_woman": p_woman,
        "p_abstain": k_abstain / n,
        "effect_p_man_minus_p_woman": effect,
        "pearson_chi2": float(pearson_chi2),
        "pearson_p": pearson_p,
        "g2": float(g2),
        "g2_p": g2_p,
        "binom_exact_p": binom_exact_p,
        "method": method,
    }


def compare_z_vs_trinomial_man_woman(
    k_man: int,
    k_woman: int,
    k_abstain: int | None = None,
    *,
    n: int | None = None,
) -> dict[str, Any]:
    """Legacy two-proportion z (wrong for within-sample) vs conditional binomial (correct)."""
    if n is None:
        if k_abstain is None:
            n = k_man + k_woman
            k_abstain = 0
        else:
            n = k_man + k_woman + k_abstain
    if k_abstain is None:
        k_abstain = max(0, n - k_man - k_woman)

    legacy_z, legacy_p, legacy_effect, legacy_ci = two_proportion_ztest(
        k_man, n, k_woman, n
    )
    tri = trinomial_choice_analysis(k_man, k_woman, k_abstain)
    pref = tri.get("preference_among_non_abstain") or {}

    alpha = 0.05
    legacy_sig = legacy_p < alpha if legacy_p == legacy_p else False
    correct_p = pref.get("p_exact", float("nan"))
    correct_sig = correct_p < alpha if correct_p == correct_p else False

    return {
        **tri,
        "legacy_two_prop_z": legacy_z,
        "legacy_two_prop_p": legacy_p,
        "legacy_two_prop_effect": legacy_effect,
        "legacy_significant_005": legacy_sig,
        "correct_z": pref.get("z"),
        "correct_p_exact": correct_p,
        "correct_p_norm": pref.get("p_norm"),
        "correct_significant_005": correct_sig,
        "same_conclusion_at_005": legacy_sig == correct_sig,
        # backward-compat keys
        "z_statistic": legacy_z,
        "z_p": legacy_p,
        "z_effect": legacy_effect,
        "z_ci_low": legacy_ci[0],
        "z_ci_high": legacy_ci[1],
        "z_significant_005": legacy_sig,
        "pearson_p": pref.get("p_exact"),
        "trinomial_significant_005": correct_sig,
    }


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


def one_way_icc(values: np.ndarray, groups: np.ndarray) -> float:
    """
    ICC(1) — one-way random effects, single measurement.

    Uses ANOVA decomposition; returns nan when not estimable.
    """
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    mask = np.isfinite(values)
    values, groups = values[mask], groups[mask]
    n = len(values)
    if n < 2:
        return float("nan")
    _, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    g = len(counts)
    if g < 2:
        return float("nan")
    grand = float(np.mean(values))
    means = np.bincount(inv, weights=values) / counts
    ssb = float(np.sum(counts * (means - grand) ** 2))
    ssw = float(np.sum((values - means[inv]) ** 2))
    df_b = g - 1
    df_w = n - g
    if df_b <= 0 or df_w <= 0:
        return float("nan")
    msb = ssb / df_b
    msw = ssw / df_w
    k_bar = float(n / g)
    denom = msb + (k_bar - 1.0) * msw
    if denom <= 0.0 or np.isclose(denom, 0.0):
        return float("nan")
    return float((msb - msw) / denom)


def cluster_one_sample_margin_stats(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    mu: float = 0.0,
) -> dict[str, Any]:
    """
    Compare naive row-level vs cluster-aggregated one-sample inference.

    Cluster unit: mean margin per group (base item). Primary test uses cluster
    means; naive stats kept for design-effect diagnostics.
    """
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    mask = np.isfinite(values)
    values, groups = values[mask], groups[mask]
    n_rows = int(len(values))
    if n_rows == 0:
        return {
            "row_n": 0,
            "n_clusters": 0,
            "k_mean": float("nan"),
            "naive_t": float("nan"),
            "naive_p": float("nan"),
            "naive_se": float("nan"),
            "cluster_t": float("nan"),
            "cluster_p": float("nan"),
            "cluster_se": float("nan"),
            "cluster_wilcoxon_stat": float("nan"),
            "cluster_wilcoxon_p": float("nan"),
            "icc": float("nan"),
            "deff": float("nan"),
            "n_eff": float("nan"),
        }

    _, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    cluster_means = np.bincount(inv, weights=values) / counts
    n_clusters = int(len(cluster_means))
    k_mean = float(n_rows / n_clusters) if n_clusters else float("nan")

    naive_t, naive_p = one_sample_ttest_vs(values, mu)
    cluster_t, cluster_p = one_sample_ttest_vs(cluster_means, mu)
    w_stat, w_p = wilcoxon_vs(cluster_means, mu)

    naive_se = float("nan")
    if n_rows >= 2:
        naive_se = float(np.std(values, ddof=1) / math.sqrt(n_rows))
    cluster_se = float("nan")
    if n_clusters >= 2:
        cluster_se = float(np.std(cluster_means, ddof=1) / math.sqrt(n_clusters))

    deff = float("nan")
    n_eff = float("nan")
    if (
        math.isfinite(naive_se)
        and math.isfinite(cluster_se)
        and naive_se > 0.0
        and cluster_se > 0.0
    ):
        deff = float((cluster_se / naive_se) ** 2)
        n_eff = float(n_rows / deff) if deff > 0.0 else float("nan")

    icc = one_way_icc(values, groups)

    return {
        "row_n": n_rows,
        "n_clusters": n_clusters,
        "k_mean": k_mean,
        "naive_t": naive_t,
        "naive_p": naive_p,
        "naive_se": naive_se,
        "cluster_t": cluster_t,
        "cluster_p": cluster_p,
        "cluster_se": cluster_se,
        "cluster_wilcoxon_stat": w_stat,
        "cluster_wilcoxon_p": w_p,
        "icc": icc,
        "deff": deff,
        "n_eff": n_eff,
    }


def cluster_choice_preference_stats(
    prefers_man: np.ndarray,
    prefers_woman: np.ndarray,
    groups: np.ndarray,
    *,
    mu_theta: float = 0.5,
) -> dict[str, Any]:
    """
    Item-level choice preference parallel to cluster_one_sample_margin_stats (prob).

    Per cluster i: M_i, W_i among gender answers; theta_i = M_i/(M_i+W_i).
    Ties (e.g. 2 man, 2 woman) yield theta_i=0.5 — no majority vote needed.

    Primary: one-sample t-test / Wilcoxon on {theta_i} vs mu_theta (default 0.5).
    Pooled conditional binomial on sum(M), sum(W) kept for comparison (row-level).
    """
    prefers_man = np.asarray(prefers_man, dtype=bool)
    prefers_woman = np.asarray(prefers_woman, dtype=bool)
    groups = np.asarray(groups)

    gender_row = prefers_man | prefers_woman
    prefers_man = prefers_man[gender_row]
    prefers_woman = prefers_woman[gender_row]
    groups = groups[gender_row]

    row_y = prefers_man.astype(float)  # 1=man, 0=woman among gender rows
    n_rows = int(len(row_y))
    if n_rows == 0:
        return {
            "row_n": 0,
            "n_clusters": 0,
            "n_clusters_with_gender": 0,
            "k_man_total": 0,
            "k_woman_total": 0,
            "pooled_theta": float("nan"),
            "pooled_z": float("nan"),
            "pooled_p_exact": float("nan"),
            "mean_theta": float("nan"),
            "mean_gap": float("nan"),
            "cluster_t": float("nan"),
            "cluster_p": float("nan"),
            "cluster_se": float("nan"),
            "cluster_wilcoxon_stat": float("nan"),
            "cluster_wilcoxon_p": float("nan"),
            "naive_t": float("nan"),
            "naive_p": float("nan"),
            "sign_n_man_wins": 0,
            "sign_n_woman_wins": 0,
            "sign_n_ties": 0,
            "sign_p_exact": float("nan"),
            "icc": float("nan"),
            "deff": float("nan"),
            "n_eff": float("nan"),
            "h0": f"P(man | man or woman) = {mu_theta} at item level (mean theta_i)",
        }

    k_man_total = int(prefers_man.sum())
    k_woman_total = int(prefers_woman.sum())
    pooled = gender_preference_test(k_man_total, k_woman_total)

    _, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    m_i = np.bincount(inv, weights=prefers_man.astype(float))
    w_i = np.bincount(inv, weights=prefers_woman.astype(float))
    n_gender_i = m_i + w_i
    has_gender = n_gender_i > 0
    theta_i = np.full(len(m_i), np.nan, dtype=float)
    theta_i[has_gender] = m_i[has_gender] / n_gender_i[has_gender]
    gap_i = np.full(len(m_i), np.nan, dtype=float)
    gap_i[has_gender] = (m_i[has_gender] - w_i[has_gender]) / n_gender_i[has_gender]

    theta_valid = theta_i[has_gender]
    n_clusters = int(len(theta_i))
    n_clusters_with_gender = int(has_gender.sum())

    man_wins = int(np.sum(m_i[has_gender] > w_i[has_gender]))
    woman_wins = int(np.sum(w_i[has_gender] > m_i[has_gender]))
    ties = int(np.sum(np.isclose(m_i[has_gender], w_i[has_gender])))
    sign_decided = man_wins + woman_wins
    if sign_decided > 0:
        sign_p = float(
            stats.binomtest(man_wins, sign_decided, 0.5, alternative="two-sided").pvalue
        )
    else:
        sign_p = float("nan")

    cluster_t, cluster_p = one_sample_ttest_vs(theta_valid, mu_theta)
    w_stat, w_p = wilcoxon_vs(theta_valid, mu_theta)
    naive_t, naive_p = one_sample_ttest_vs(row_y, mu_theta)

    cluster_se = float("nan")
    if n_clusters_with_gender >= 2:
        cluster_se = float(np.std(theta_valid, ddof=1) / math.sqrt(n_clusters_with_gender))
    naive_se = float("nan")
    if n_rows >= 2:
        naive_se = float(np.std(row_y, ddof=1) / math.sqrt(n_rows))

    deff = float("nan")
    n_eff = float("nan")
    if (
        math.isfinite(naive_se)
        and math.isfinite(cluster_se)
        and naive_se > 0.0
        and cluster_se > 0.0
    ):
        deff = float((cluster_se / naive_se) ** 2)
        n_eff = float(n_rows / deff) if deff > 0.0 else float("nan")

    icc = one_way_icc(row_y, groups)
    mean_theta = float(np.mean(theta_valid)) if len(theta_valid) else float("nan")
    mean_gap = float(np.mean(gap_i[has_gender])) if has_gender.any() else float("nan")
    ci_theta = (float("nan"), float("nan"))
    if n_clusters_with_gender >= 2 and math.isfinite(cluster_se):
        ci_theta = (mean_theta - 1.96 * cluster_se, mean_theta + 1.96 * cluster_se)

    return {
        "row_n": n_rows,
        "n_clusters": n_clusters,
        "n_clusters_with_gender": n_clusters_with_gender,
        "k_man_total": k_man_total,
        "k_woman_total": k_woman_total,
        "pooled_theta": pooled["theta_hat"],
        "pooled_conditional_gap": pooled["conditional_gap"],
        "pooled_z": pooled["z"],
        "pooled_p_exact": pooled["p_exact"],
        "pooled_p_norm": pooled["p_norm"],
        "mean_theta": mean_theta,
        "mean_gap": mean_gap,
        "cluster_t": cluster_t,
        "cluster_p": cluster_p,
        "cluster_se": cluster_se,
        "ci_theta_cluster_wald_95": ci_theta,
        "cluster_wilcoxon_stat": w_stat,
        "cluster_wilcoxon_p": w_p,
        "naive_t": naive_t,
        "naive_p": naive_p,
        "naive_se": naive_se,
        "sign_n_man_wins": man_wins,
        "sign_n_woman_wins": woman_wins,
        "sign_n_ties": ties,
        "sign_p_exact": sign_p,
        "icc": icc,
        "deff": deff,
        "n_eff": n_eff,
        "h0": f"E[theta_i] = {mu_theta}, theta_i = M_i/(M_i+W_i) per cluster",
        "tie_rule": "theta_i=0.5 when M_i=W_i (no majority vote)",
    }


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
