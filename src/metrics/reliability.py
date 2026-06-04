"""Heuristic reliability flags for behavioral test results."""

from __future__ import annotations

import math
from typing import Any

from src.metrics.stats_tests import TestResult

RELIABILITY_OK = "ok"
RELIABILITY_CAUTION = "caution"
RELIABILITY_UNRELIABLE = "unreliable"
RELIABILITY_NOT_RUN = "not_run"

N_UNRELIABLE = 10
N_CAUTION = 30
MIN_CELL_UNRELIABLE = 1
MIN_CELL_CAUTION = 10
CHI2_MIN_OBS_UNRELIABLE = 0
CHI2_MIN_EXP_CAUTION = 5
MCNEMAR_DISC_UNRELIABLE = 10
MCNEMAR_DISC_CAUTION = 25


def _as_int(x: Any) -> int | None:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return int(x)


def _min_cell(k: int | None, n: int | None) -> int | None:
    if k is None or n is None or n <= 0:
        return None
    return min(k, n - k)


def _add(reasons: list[str], status: str, new: str, level: str) -> str:
    reasons.append(new)
    if level == RELIABILITY_UNRELIABLE:
        return RELIABILITY_UNRELIABLE
    if status == RELIABILITY_UNRELIABLE:
        return status
    if level == RELIABILITY_CAUTION and status == RELIABILITY_OK:
        return RELIABILITY_CAUTION
    return status


def _check_proportion_cells(
    reasons: list[str],
    status: str,
    *,
    k: int | None,
    n: int | None,
    label: str,
) -> str:
    n_i = _as_int(n)
    k_i = _as_int(k)
    if n_i is None:
        return status
    if n_i < N_UNRELIABLE:
        status = _add(reasons, status, f"{label}n={n_i}<{N_UNRELIABLE}", RELIABILITY_UNRELIABLE)
    elif n_i < N_CAUTION:
        status = _add(reasons, status, f"{label}n={n_i}<{N_CAUTION}", RELIABILITY_CAUTION)

    mc = _min_cell(k_i, n_i)
    if mc is not None:
        if mc < MIN_CELL_UNRELIABLE:
            status = _add(
                reasons,
                status,
                f"{label}min(k,n-k)={mc}<{MIN_CELL_UNRELIABLE}",
                RELIABILITY_UNRELIABLE,
            )
        elif mc < MIN_CELL_CAUTION:
            status = _add(
                reasons,
                status,
                f"{label}min(k,n-k)={mc}<{MIN_CELL_CAUTION}",
                RELIABILITY_CAUTION,
            )

    if k_i is not None and n_i > 0:
        p = k_i / n_i
        if p in (0.0, 1.0) and n_i >= N_UNRELIABLE:
            status = _add(
                reasons,
                status,
                f"{label}p_hat={p:.3f} насыщение",
                RELIABILITY_UNRELIABLE,
            )
        elif (p <= 0.05 or p >= 0.95) and n_i >= N_UNRELIABLE:
            status = _add(
                reasons,
                status,
                f"{label}p_hat={p:.3f} экстремальная доля",
                RELIABILITY_CAUTION,
            )
    return status


def _parse_chi2_table(r: TestResult) -> list[list[float]] | None:
    table = r.extra.get("table")
    if table is None:
        return None
    if isinstance(table, str):
        import ast

        try:
            table = ast.literal_eval(table)
        except (SyntaxError, ValueError):
            return None
    if not isinstance(table, list):
        return None
    return table


def assess_reliability(r: TestResult) -> tuple[str, list[str]]:
    if r.extra.get("status") == "skipped":
        return RELIABILITY_NOT_RUN, ["тест не выполнялся (skipped)"]

    if r.p_raw is not None and isinstance(r.p_raw, float) and math.isnan(r.p_raw):
        if r.extra.get("note"):
            return RELIABILITY_OK, []
        return RELIABILITY_UNRELIABLE, ["p_value отсутствует (NaN)"]

    reasons: list[str] = []
    status = RELIABILITY_OK
    tid = r.test_id

    n_eff = _as_int(r.extra.get("n_eff"))
    n1 = _as_int(r.n1)
    n_use = n_eff if n_eff is not None else n1

    if "mcnemar" in tid:
        b = _as_int(r.extra.get("discordant_b_man_yes_woman_no")) or _as_int(r.k1) or 0
        c = _as_int(r.extra.get("discordant_c_man_no_woman_yes")) or _as_int(r.k2) or 0
        disc = b + c
        if disc < MCNEMAR_DISC_UNRELIABLE:
            status = _add(
                reasons,
                status,
                f"дискордантных пар={disc}<{MCNEMAR_DISC_UNRELIABLE}",
                RELIABILITY_UNRELIABLE,
            )
        elif disc < MCNEMAR_DISC_CAUTION:
            status = _add(
                reasons,
                status,
                f"дискордантных пар={disc}<{MCNEMAR_DISC_CAUTION}",
                RELIABILITY_CAUTION,
            )
        if disc > 0 and (b == 0 or c == 0):
            status = _add(
                reasons,
                status,
                f"односторонняя дискордантность b={b}, c={c}",
                RELIABILITY_CAUTION,
            )
        return status, reasons

    if "chi2" in tid:
        table = _parse_chi2_table(r)
        if table:
            obs_flat = [int(x) for row in table for x in row]
            if any(x <= CHI2_MIN_OBS_UNRELIABLE for x in obs_flat):
                status = _add(
                    reasons,
                    status,
                    "χ²: наблюдаемая ячейка = 0",
                    RELIABILITY_UNRELIABLE,
                )
            if any(0 < x < MIN_CELL_UNRELIABLE for x in obs_flat):
                status = _add(
                    reasons,
                    status,
                    f"χ²: ячейка < {MIN_CELL_UNRELIABLE}",
                    RELIABILITY_CAUTION,
                )
        expected = r.extra.get("expected")
        if expected is not None and not isinstance(expected, str):
            try:
                exp_flat = [float(x) for row in expected for x in row]
                if any(x < 1 for x in exp_flat):
                    status = _add(
                        reasons,
                        status,
                        "χ²: ожидаемая частота < 1",
                        RELIABILITY_UNRELIABLE,
                    )
                elif any(x < CHI2_MIN_EXP_CAUTION for x in exp_flat):
                    status = _add(
                        reasons,
                        status,
                        f"χ²: ожидаемая частота < {CHI2_MIN_EXP_CAUTION}",
                        RELIABILITY_CAUTION,
                    )
            except (TypeError, ValueError):
                pass
        if r.p_raw == 1.0 and r.statistic == 0.0:
            status = _add(
                reasons,
                status,
                "χ²: нулевая статистика (нет вариации по категориям)",
                RELIABILITY_UNRELIABLE,
            )
        return status, reasons

    if r.n2 is not None and r.k2 is not None:
        n1_use = n_eff if n_eff is not None else n1
        n2 = _as_int(r.extra.get("n2_eff")) or _as_int(r.n2)
        status = _check_proportion_cells(reasons, status, k=_as_int(r.k1), n=n1_use, label="группа1 ")
        status = _check_proportion_cells(reasons, status, k=_as_int(r.k2), n=n2, label="группа2 ")
        if r.statistic is not None and isinstance(r.statistic, float) and math.isnan(r.statistic):
            status = _add(reasons, status, "z=NaN (нулевая дисперсия)", RELIABILITY_UNRELIABLE)
        return status, reasons

    if r.p1 is not None and r.k1 is None and n_use is not None:
        p1 = float(r.p1)
        if p1 in (0.0, 1.0):
            status = _add(
                reasons,
                status,
                f"mean rate={p1:.3f} насыщение (family-level)",
                RELIABILITY_UNRELIABLE,
            )
        elif p1 <= 0.05 or p1 >= 0.95:
            status = _add(
                reasons,
                status,
                f"mean rate={p1:.3f} экстремальная",
                RELIABILITY_CAUTION,
            )
        return status, reasons

    if r.k1 is not None and n_use is not None:
        status = _check_proportion_cells(reasons, status, k=_as_int(r.k1), n=n_use, label="")
        if "3opt" in tid and _as_int(r.k1) is not None and _as_int(r.k1) < 10:
            status = _add(
                reasons,
                status,
                f"k_stereo={int(r.k1)}<10 (массовый abstain)",
                RELIABILITY_UNRELIABLE,
            )
        return status, reasons

    return status, reasons


def apply_reliability_labels(results: list[TestResult]) -> None:
    for r in results:
        level, reasons = assess_reliability(r)
        r.extra["reliability"] = level
        r.extra["reliability_reasons"] = reasons


def reliability_badge(level: str | None) -> str:
    if level == RELIABILITY_UNRELIABLE:
        return " ⚠ **Данные ненадёжные**"
    if level == RELIABILITY_CAUTION:
        return " ⚡ *осторожно*"
    if level == RELIABILITY_NOT_RUN:
        return " — _не запускался_"
    return ""


def reliability_reason_suffix(r: TestResult) -> str:
    reasons = r.extra.get("reliability_reasons") or []
    if not reasons:
        return ""
    return " — " + "; ".join(reasons)
