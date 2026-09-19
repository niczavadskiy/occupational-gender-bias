"""
Stratified contrast PCA: is live gender preference one direction or a subspace?

Rows of D are per-stratum mean-diffs, not per-row HS:

    d_s = mean(h | s, y=M) − mean(h | s, y=F)

Uncentered SVD of D recovers the pooled contrast as PC1 (should match v_G).
Centered PCA of *unit* rows tests angular heterogeneity: do strata point
along the same axis, or fan out?

Shuffle-null: permute y inside each stratum (class counts preserved), rebuild D.
A component is "above null" if its explained-variance share exceeds the 95th
percentile of the null spectrum at that index.

Steering basis: V_k = [v_G, PC1_⊥, …] after Gram–Schmidt on the unit-centered
right singular vectors. Stage A should still sweep k even if the diagnostic
says rank 1 — that bake-off is the causal test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EPS = 1e-12


def unit(v: np.ndarray, eps: float = EPS) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    n = float(np.linalg.norm(v))
    if n < eps:
        raise ValueError("zero vector")
    return v / n


def cosine(a: np.ndarray, b: np.ndarray, eps: float = EPS) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return float("nan")
    return float((a / na) @ (b / nb))


def participation_ratio(eigs: np.ndarray) -> float:
    lam = np.asarray(eigs, dtype=np.float64)
    lam = lam[np.isfinite(lam) & (lam > 0)]
    if lam.size == 0:
        return float("nan")
    s1 = float(lam.sum())
    s2 = float((lam * lam).sum())
    if s2 < EPS:
        return float("nan")
    return (s1 * s1) / s2


def mean_diff(H: np.ndarray, y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    pos, neg = H[y == 1], H[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("mean_diff needs both classes")
    return unit(pos.mean(axis=0) - neg.mean(axis=0))


@dataclass
class StratumDiffs:
    D: np.ndarray
    names: list[str]
    n_pos: list[int]
    n_neg: list[int]
    n_dropped: int
    dropped: list[str] = field(default_factory=list)

    @property
    def n_strata(self) -> int:
        return int(self.D.shape[0]) if self.D.size else 0


def stratum_diffs(
    H: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    *,
    min_per_class: int = 1,
) -> StratumDiffs:
    """Per-label mean-diff on rows where mask is true. Skip unbalanced strata."""
    H = np.asarray(H, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    labels = np.asarray(labels)
    mask = np.asarray(mask, dtype=bool)
    rows: list[np.ndarray] = []
    names: list[str] = []
    n_pos: list[int] = []
    n_neg: list[int] = []
    dropped: list[str] = []
    for name in sorted(set(labels.tolist()), key=str):
        m = mask & (labels == name)
        yi, Hi = y[m], H[m]
        np_ = int((yi == 1).sum())
        nn_ = int((yi == 0).sum())
        if np_ < min_per_class or nn_ < min_per_class:
            dropped.append(f"{name} (M={np_}, F={nn_})")
            continue
        d = Hi[yi == 1].mean(axis=0) - Hi[yi == 0].mean(axis=0)
        rows.append(d)
        names.append(str(name))
        n_pos.append(np_)
        n_neg.append(nn_)
    if not rows:
        d = int(H.shape[1])
        return StratumDiffs(
            D=np.zeros((0, d), dtype=np.float64),
            names=[],
            n_pos=[],
            n_neg=[],
            n_dropped=len(dropped),
            dropped=dropped,
        )
    return StratumDiffs(
        D=np.stack(rows, axis=0),
        names=names,
        n_pos=n_pos,
        n_neg=n_neg,
        n_dropped=len(dropped),
        dropped=dropped,
    )


@dataclass
class Spectrum:
    singular_values: np.ndarray
    V: np.ndarray  # (d, k) right singular vectors
    explained: np.ndarray
    cumulative: np.ndarray
    participation_ratio: float
    centered: bool
    row_unit: bool
    mean: np.ndarray

    def as_dict(self, *, n_keep: int = 12) -> dict:
        k = min(int(n_keep), int(self.explained.size))
        return {
            "centered": self.centered,
            "row_unit": self.row_unit,
            "n_components": int(self.explained.size),
            "explained": [float(x) for x in self.explained[:k]],
            "cumulative": [float(x) for x in self.cumulative[:k]],
            "singular_values": [float(x) for x in self.singular_values[:k]],
            "participation_ratio": float(self.participation_ratio),
            "mean_norm": float(np.linalg.norm(self.mean)),
        }


def svd_spectrum(
    D: np.ndarray,
    *,
    center: bool,
    row_unit: bool = False,
) -> Spectrum:
    D = np.asarray(D, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] == 0:
        raise ValueError("D must be (n_strata, d) with n_strata ≥ 1")
    X = D.copy()
    if row_unit:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        good = norms.ravel() >= EPS
        X = X[good]
        norms = norms[good]
        if X.shape[0] == 0:
            raise ValueError("all stratum diffs were zero")
        X = X / np.maximum(norms, EPS)
    mean = X.mean(axis=0)
    if center:
        X = X - mean
    k_max = min(X.shape)
    if k_max == 0:
        raise ValueError("empty matrix after preprocessing")
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    energy = S * S
    tot = float(energy.sum())
    explained = energy / tot if tot > EPS else np.zeros_like(energy)
    cumulative = np.cumsum(explained)
    return Spectrum(
        singular_values=S,
        V=Vt.T,
        explained=explained,
        cumulative=cumulative,
        participation_ratio=participation_ratio(energy),
        centered=center,
        row_unit=row_unit,
        mean=mean,
    )


def shuffle_y_within_strata(
    y: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    y_null = np.asarray(y, dtype=int).copy()
    labels = np.asarray(labels)
    mask = np.asarray(mask, dtype=bool)
    for name in set(labels[mask].tolist()):
        idx = np.flatnonzero(mask & (labels == name))
        y_null[idx] = rng.permutation(y_null[idx])
    return y_null


def null_spectra(
    H: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    *,
    n_null: int,
    seed: int,
    min_per_class: int,
    center: bool,
    row_unit: bool,
) -> dict:
    """Mean / p95 explained-variance curves under within-stratum y-shuffle."""
    rng = np.random.default_rng(seed)
    curves: list[np.ndarray] = []
    prs: list[float] = []
    max_k = 0
    for _ in range(n_null):
        y_n = shuffle_y_within_strata(y, labels, mask, rng)
        diff = stratum_diffs(H, y_n, labels, mask, min_per_class=min_per_class)
        if diff.n_strata < 2:
            continue
        spec = svd_spectrum(diff.D, center=center, row_unit=row_unit)
        curves.append(spec.explained)
        prs.append(spec.participation_ratio)
        max_k = max(max_k, spec.explained.size)
    if not curves:
        return {
            "n_ok": 0,
            "mean_explained": [],
            "p95_explained": [],
            "pr_mean": float("nan"),
            "pr_p95": float("nan"),
        }
    stacked = np.full((len(curves), max_k), np.nan, dtype=np.float64)
    for i, c in enumerate(curves):
        stacked[i, : c.size] = c
    return {
        "n_ok": len(curves),
        "mean_explained": [float(x) for x in np.nanmean(stacked, axis=0)],
        "p95_explained": [float(x) for x in np.nanpercentile(stacked, 95, axis=0)],
        "pr_mean": float(np.nanmean(prs)),
        "pr_p95": float(np.nanpercentile(prs, 95)),
    }


def n_above_null(
    observed: np.ndarray,
    p95: list[float] | np.ndarray,
    *,
    margin: float = 0.01,
) -> int:
    obs = np.asarray(observed, dtype=np.float64)
    ref = np.asarray(p95, dtype=np.float64)
    k = min(obs.size, ref.size)
    n = 0
    for i in range(k):
        if obs[i] > ref[i] + margin:
            n += 1
        else:
            break
    return n


def orthonormalize(cols: list[np.ndarray], *, k_max: int, eps: float = 1e-6) -> np.ndarray:
    kept: list[np.ndarray] = []
    for raw in cols:
        w = np.asarray(raw, dtype=np.float64).ravel()
        for c in kept:
            w = w - float(w @ c) * c
        n = float(np.linalg.norm(w))
        if n < eps:
            continue
        kept.append(w / n)
        if len(kept) >= k_max:
            break
    if not kept:
        raise ValueError("no linearly independent columns")
    return np.stack(kept, axis=1)


def build_steering_basis(v_g: np.ndarray, V_extra: np.ndarray, k_max: int) -> np.ndarray:
    """V_k = [v_G, extra PCs after Gram–Schmidt]. Extra is (d, m)."""
    cols = [unit(v_g)]
    extra = np.asarray(V_extra, dtype=np.float64)
    if extra.ndim == 1:
        extra = extra[:, None]
    for j in range(extra.shape[1]):
        cols.append(extra[:, j])
    return orthonormalize(cols, k_max=k_max)


def pairwise_vs_vg(diff: StratumDiffs, v_g: np.ndarray) -> list[dict]:
    out = []
    for i, name in enumerate(diff.names):
        out.append(
            {
                "stratum": name,
                "n_man": diff.n_pos[i],
                "n_woman": diff.n_neg[i],
                "norm": float(np.linalg.norm(diff.D[i])),
                "cos_vg": cosine(diff.D[i], v_g),
            }
        )
    return out


def interpret(
    *,
    cos_uncent_pc1_vg: float,
    uncent_pc1_share: float,
    n_above_unit_centered: int,
    unit_pr: float,
    unit_pr_null_p95: float,
    n_strata: int,
) -> dict:
    """Geometry-only verdict. Causal test is Stage A k vs 1."""
    pc1_ok = abs(cos_uncent_pc1_vg) >= 0.9 and uncent_pc1_share >= 0.45
    pr_1d = np.isfinite(unit_pr) and (
        not np.isfinite(unit_pr_null_p95) or unit_pr <= unit_pr_null_p95 + 0.35
    )
    if n_strata < 4:
        verdict = "underpowered"
        note = (
            f"мало страт с обоими классами (n={n_strata}); "
            "спектр шумный — не отвергать 1D"
        )
    elif n_above_unit_centered == 0 and pc1_ok:
        verdict = "one_direction"
        note = (
            "uncentered PC1 ≈ v_G, угловой спектр не выше shuffle-null — "
            "гипотеза одного gender direction жива"
        )
    elif 1 <= n_above_unit_centered <= 3:
        verdict = "low_rank_subspace"
        note = (
            f"{n_above_unit_centered} угловые компонент(ы) над null "
            f"(effective rank ≈ {1 + n_above_unit_centered}) — "
            "имеет смысл Stage A bake-off V_k vs v_G"
        )
    else:
        verdict = "high_rank_or_noise"
        extra = "" if pr_1d else "; participation ratio тоже выше null"
        note = (
            "длинный спектр на уровне шума или много компонент над null; "
            "проверить сборку D (slot/occupation leakage)" + extra
        )
    return {
        "verdict": verdict,
        "n_above_unit_centered": int(n_above_unit_centered),
        "pc1_aligns_vg": bool(pc1_ok),
        "note": note,
    }


def layer_report(
    H: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    *,
    soc: np.ndarray,
    context_order: np.ndarray,
    position_variant: np.ndarray,
    v_g: np.ndarray,
    n_null: int,
    seed: int,
    min_per_class: int,
    k_max: int,
    probes: dict[str, np.ndarray] | None = None,
) -> dict:
    """Full diagnostic + steering basis for one layer."""
    soc_diff = stratum_diffs(H, y, soc, train, min_per_class=min_per_class)
    layout_diff = stratum_diffs(H, y, context_order, train, min_per_class=min_per_class)
    pos_diff = stratum_diffs(H, y, position_variant, train, min_per_class=min_per_class)

    if soc_diff.n_strata < 2:
        return {
            "n_soc_strata": soc_diff.n_strata,
            "soc_dropped": soc_diff.dropped,
            "error": "need ≥2 SOC strata with both classes on train",
        }

    uncent = svd_spectrum(soc_diff.D, center=False, row_unit=False)
    centered = svd_spectrum(soc_diff.D, center=True, row_unit=False)
    unit_c = svd_spectrum(soc_diff.D, center=True, row_unit=True)
    null = null_spectra(
        H,
        y,
        soc,
        train,
        n_null=n_null,
        seed=seed,
        min_per_class=min_per_class,
        center=True,
        row_unit=True,
    )
    n_above = n_above_null(unit_c.explained, null.get("p95_explained") or [])
    k_write = max(1, min(k_max, soc_diff.n_strata, int(H.shape[1])))
    W = build_steering_basis(v_g, unit_c.V, k_write)

    cos_pc1 = cosine(uncent.V[:, 0], v_g)
    # SVD sign is arbitrary
    if np.isfinite(cos_pc1) and cos_pc1 < 0:
        uncent.V[:, 0] *= -1
        cos_pc1 = -cos_pc1

    interp = interpret(
        cos_uncent_pc1_vg=cos_pc1,
        uncent_pc1_share=float(uncent.explained[0]),
        n_above_unit_centered=n_above,
        unit_pr=unit_c.participation_ratio,
        unit_pr_null_p95=float(null.get("pr_p95", float("nan"))),
        n_strata=soc_diff.n_strata,
    )

    probe_on_basis: list[dict] = []
    for j in range(min(6, W.shape[1])):
        row = {"j": j, "role": "v_G" if j == 0 else f"pc_orth_{j}", "cos_vg": cosine(W[:, j], v_g)}
        for name, vec in (probes or {}).items():
            row[f"cos_{name}"] = cosine(W[:, j], vec)
        probe_on_basis.append(row)

    return {
        "n_soc_strata": soc_diff.n_strata,
        "soc_dropped": soc_diff.dropped,
        "soc_rows": pairwise_vs_vg(soc_diff, v_g),
        "layout_rows": pairwise_vs_vg(layout_diff, v_g),
        "position_rows": pairwise_vs_vg(pos_diff, v_g),
        "uncentered": uncent.as_dict(),
        "centered": centered.as_dict(),
        "unit_centered": unit_c.as_dict(),
        "cos_uncent_pc1_vg": cos_pc1,
        "null_unit_centered": null,
        "n_above_null": n_above,
        "interpret": interp,
        "k_found": int(W.shape[1]),
        "basis_cos": probe_on_basis,
        "W": W,
    }


def report_to_jsonable(rep: dict) -> dict:
    out = {}
    for k, v in rep.items():
        if k.startswith("_") or k == "W":
            continue
        out[k] = v
    return out


def spectrum_table_md(
    title: str,
    explained: list[float],
    cumulative: list[float],
    p95: list[float] | None = None,
    n_rows: int = 8,
) -> str:
    lines = [f"### {title}", "", "| PC | share | cum | null p95 | above |", "| ---: | ---: | ---: | ---: | :---: |"]
    k = min(n_rows, len(explained))
    for i in range(k):
        share = explained[i]
        cum = cumulative[i] if i < len(cumulative) else float("nan")
        if p95 is not None and i < len(p95):
            ref = p95[i]
            mark = "yes" if share > ref + 0.01 else ""
            ref_s = f"{ref:.3f}"
        else:
            mark, ref_s = "", "—"
        lines.append(f"| {i + 1} | {share:.3f} | {cum:.3f} | {ref_s} | {mark} |")
    return "\n".join(lines) + "\n"


def layer_markdown(layer: int, rep: dict) -> str:
    if "error" in rep:
        return f"## L{layer}\n\n**ошибка:** {rep['error']}\n\nдроп: {', '.join(rep.get('soc_dropped') or []) or '—'}\n"

    interp = rep["interpret"]
    un = rep["uncentered"]
    uc = rep["unit_centered"]
    null = rep["null_unit_centered"]
    lines = [
        f"## L{layer}",
        "",
        f"**вердикт:** `{interp['verdict']}` — {interp['note']}",
        "",
        f"- SOC страт с обоими классами: **{rep['n_soc_strata']}** (дроп {len(rep.get('soc_dropped') or [])})",
        f"- uncentered PC1 share = {un['explained'][0]:.3f}, cos(PC1, v_G) = {rep['cos_uncent_pc1_vg']:+.3f}",
        f"- unit-centered PR = {uc['participation_ratio']:.2f} (null p95 {null.get('pr_p95', float('nan')):.2f})",
        f"- компонент над null: **{rep['n_above_null']}**; k_found (базис) = {rep['k_found']}",
        "",
        spectrum_table_md(
            "Uncentered SVD (сырой D)",
            un["explained"],
            un["cumulative"],
        ),
        spectrum_table_md(
            "Unit-row centered PCA (угловая гетерогенность) + shuffle-null",
            uc["explained"],
            uc["cumulative"],
            null.get("p95_explained"),
        ),
        "### cos(d_s, v_G) по SOC",
        "",
        "| SOC | n_M | n_F | ‖d‖ | cos v_G |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for row in rep.get("soc_rows") or []:
        short = row["stratum"].replace(" Occupations", "")
        lines.append(
            f"| {short} | {row['n_man']} | {row['n_woman']} | {row['norm']:.3f} | {row['cos_vg']:+.3f} |"
        )
    lines += ["", "### layout / position vs v_G", ""]
    for kind, key in (("context_order", "layout_rows"), ("position_variant", "position_rows")):
        lines.append(f"**{kind}**")
        lines.append("")
        lines.append("| stratum | n_M | n_F | cos v_G |")
        lines.append("| :--- | ---: | ---: | ---: |")
        for row in rep.get(key) or []:
            lines.append(
                f"| {row['stratum']} | {row['n_man']} | {row['n_woman']} | {row['cos_vg']:+.3f} |"
            )
        lines.append("")
    if rep.get("basis_cos"):
        lines += ["### базис V_k vs v_G / probe", "", "| j | role | cos v_G | extra |", "| ---: | :--- | ---: | :--- |"]
        for row in rep["basis_cos"]:
            extra = ", ".join(
                f"{k}={v:+.3f}"
                for k, v in row.items()
                if k.startswith("cos_") and k != "cos_vg" and isinstance(v, float) and np.isfinite(v)
            )
            lines.append(f"| {row['j']} | {row['role']} | {row['cos_vg']:+.3f} | {extra or '—'} |")
        lines.append("")
    return "\n".join(lines)

