"""
Unit-тесты rank-k интервенции (§30). CPU, модель не грузится.

  T1 orthonormal basis   WᵀW ≈ I для всех сохранённых подпространств
  T2 projection removal   project_out → Wᵀh_⊥ ≈ 0
  T3 center α=1           Wᵀh' = c
  T4 α=0                  h' == h побитово (и hook не трогает другие позиции)
  T5 rank-1 compatibility SubspaceSpec(k=1) == InterventionSpec(center) — regression

T1 читает steering/subspaces/inlp_<target>_<ver>.npz; T2–T5 работают на
синтетических данных и на реальном W, если файл есть. Хук проверяется на
игрушечном nn.Module с тем же интерфейсом, что декодер (ModuleList "layers").

Запуск из корня репозитория:
    python -m steering.check_inlp
    python -m steering.check_inlp --subspaces steering/subspaces/inlp_gender_choice_smoke.npz

Код 0 — все тесты в допуске; 1 — провал.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

from steering.intervene import (
    InterventionSpec,
    ProjectionTrace,
    SubspaceSpec,
    SubspaceTrace,
    _apply,
    _apply_subspace,
    steered,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
STEERING_DIR = Path(__file__).resolve().parent
DEFAULT_SUBSPACES = STEERING_DIR / "subspaces" / "inlp_gender_choice_v1.npz"

TOL_ORTHO = 1e-4
TOL_PROJECTION = 1e-4
TOL_RANK1 = 1e-6


class _ToyConfig:
    num_hidden_layers = 2
    hidden_size = 8


class _ToyBlock(nn.Module):
    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor]:
        return (hidden,)


class _ToyModel(nn.Module):
    """Минимальный носитель хука: ModuleList "layers" + embed_tokens."""

    def __init__(self, d: int = 8) -> None:
        super().__init__()
        self.config = _ToyConfig()
        self.config.hidden_size = d
        self.embed_tokens = nn.Embedding(4, d)
        self.layers = nn.ModuleList([_ToyBlock(), _ToyBlock()])

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for block in self.layers:
            hidden = block(hidden)[0]
        return hidden


def load_subspaces(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        return {}
    with np.load(path) as z:
        return {k: np.asarray(z[k], dtype=np.float32) for k in z.files}


def apply_numpy(W: np.ndarray, c: np.ndarray, h: np.ndarray, *, alpha: float, kind: str) -> np.ndarray:
    s = W.T @ h
    delta_s = -alpha * (s - c) if kind == "center" else -s
    return h + W @ delta_s


def t1_orthonormal(arrays: dict[str, np.ndarray]) -> dict:
    """WᵀW ≈ I для каждого сохранённого базиса (включая random control)."""
    rows = []
    for key in sorted(k for k in arrays if k.endswith("W") or "W_random" in k):
        W = arrays[key]
        if W.ndim != 2 or W.shape[1] == 0:
            continue
        G = W.T @ W
        err = float(np.max(np.abs(G - np.eye(G.shape[0], dtype=W.dtype))))
        rows.append({"key": key, "rank": int(W.shape[1]), "max_abs_error": err, "ok": err <= TOL_ORTHO})
    return {
        "name": "T1 orthonormal basis",
        "tolerance": TOL_ORTHO,
        "n_checked": len(rows),
        "passed": all(r["ok"] for r in rows) if rows else None,
        "detail": rows,
    }


def t2_projection_removal(bases: list[tuple[str, np.ndarray]], rng: np.random.Generator) -> dict:
    """project_out убирает подпространство целиком: Wᵀh_⊥ ≈ 0."""
    rows = []
    for name, W in bases:
        d, k = W.shape
        h = rng.normal(scale=12.0 / np.sqrt(d), size=d).astype(np.float64)
        c = np.zeros(k)
        h_perp = apply_numpy(W.astype(np.float64), c, h, alpha=1.0, kind="project_out")
        err = float(np.max(np.abs(W.astype(np.float64).T @ h_perp)))
        rows.append({"basis": name, "rank": k, "max_abs_projection": err, "ok": err <= TOL_PROJECTION})
    return {
        "name": "T2 projection removal",
        "tolerance": TOL_PROJECTION,
        "passed": all(r["ok"] for r in rows),
        "detail": rows,
    }


def t3_center_alpha1(bases: list[tuple[str, np.ndarray]], rng: np.random.Generator) -> dict:
    """При α=1 каждая координата подпространства садится в свой midpoint."""
    rows = []
    for name, W in bases:
        d, k = W.shape
        h = rng.normal(scale=12.0 / np.sqrt(d), size=d).astype(np.float64)
        c = rng.normal(size=k)
        W64 = W.astype(np.float64)
        h_after = apply_numpy(W64, c, h, alpha=1.0, kind="center")
        err = float(np.max(np.abs(W64.T @ h_after - c)))
        rows.append({"basis": name, "rank": k, "max_abs_error": err, "ok": err <= TOL_PROJECTION})

    # Тот же тест через хук в float32 (реальный путь исполнения).
    hook_rows = []
    for name, W in bases:
        d, k = W.shape
        model = _ToyModel(d)
        c = rng.normal(size=k).astype(np.float32)
        spec = SubspaceSpec(layer=1, W=np.ascontiguousarray(W), c=c, kind="center", alpha=1.0)
        hidden = torch.randn(1, 5, d) * (12.0 / np.sqrt(d))
        trace = SubspaceTrace()
        with steered(model, [spec], trace):
            out = model(hidden)
        s_after = trace.after[1]
        err = float(np.max(np.abs(s_after - c)))
        recomputed = float(
            np.max(np.abs(np.ascontiguousarray(W).T @ out[0, -1].numpy() - c))
        )
        hook_rows.append(
            {
                "basis": name,
                "rank": k,
                "max_abs_error_trace": err,
                "max_abs_error_recomputed": recomputed,
                "ok": max(err, recomputed) <= TOL_PROJECTION,
            }
        )

    return {
        "name": "T3 center alpha=1 → Wᵀh' = c",
        "tolerance": TOL_PROJECTION,
        "passed": all(r["ok"] for r in rows + hook_rows),
        "detail_numpy": rows,
        "detail_hook_float32": hook_rows,
    }


def t4_alpha_zero(bases: list[tuple[str, np.ndarray]], rng: np.random.Generator) -> dict:
    """α=0 — тождественное преобразование; остальные позиции не тронуты вообще."""
    rows = []
    for name, W in bases:
        d, k = W.shape
        model = _ToyModel(d)
        c = rng.normal(size=k).astype(np.float32)
        spec = SubspaceSpec(layer=1, W=np.ascontiguousarray(W), c=c, kind="center", alpha=0.0)
        hidden = torch.randn(1, 5, d) * (12.0 / np.sqrt(d))
        with steered(model, [spec]):
            out = model(hidden)
        last_identical = bool(torch.equal(out[0, -1], hidden[0, -1]))
        rest_identical = bool(torch.equal(out[0, :-1], hidden[0, :-1]))
        rows.append(
            {
                "basis": name,
                "rank": k,
                "last_token_bitwise_identical": last_identical,
                "other_positions_untouched": rest_identical,
                "ok": last_identical and rest_identical,
            }
        )
    return {"name": "T4 alpha=0 → h' = h", "passed": all(r["ok"] for r in rows), "detail": rows}


def t5_rank1_compatibility(rng: np.random.Generator, *, d: int = 64, n: int = 32) -> dict:
    """SubspaceSpec(k=1, center) должен воспроизводить InterventionSpec(center)."""
    rows = []
    for alpha in (0.25, 0.5, 1.0, 1.25):
        w = rng.normal(size=d)
        w /= np.linalg.norm(w)
        c = float(rng.normal())
        rank1 = InterventionSpec(layer=1, w=w.astype(np.float32), kind="center", alpha=alpha, c=c)
        rankk = SubspaceSpec(
            layer=1,
            W=w.astype(np.float32).reshape(d, 1),
            c=np.array([c], dtype=np.float32),
            kind="center",
            alpha=alpha,
        )
        w_t = torch.as_tensor(w, dtype=torch.float32)
        W_t = torch.as_tensor(w.reshape(d, 1), dtype=torch.float32)
        c_t = torch.as_tensor([c], dtype=torch.float32)

        max_h = 0.0
        max_s = 0.0
        for _ in range(n):
            h = torch.randn(d) * (12.0 / np.sqrt(d))
            h_a, s_b_a, s_a_a = _apply(h, rank1, w_t)
            h_b, s_b_b, s_a_b, _hn, _dn = _apply_subspace(h, rankk, W_t, c_t)
            max_h = max(max_h, float(torch.max(torch.abs(h_a - h_b))))
            max_s = max(max_s, abs(s_b_a - float(s_b_b[0])), abs(s_a_a - float(s_a_b[0])))
        rows.append(
            {
                "alpha": alpha,
                "max_abs_diff_h": max_h,
                "max_abs_diff_s": max_s,
                "ok": max(max_h, max_s) <= TOL_RANK1,
            }
        )
    return {
        "name": "T5 rank-1 compatibility",
        "tolerance": TOL_RANK1,
        "passed": all(r["ok"] for r in rows),
        "detail": rows,
    }


def synthetic_bases(rng: np.random.Generator) -> list[tuple[str, np.ndarray]]:
    out = []
    for d, k in ((64, 1), (64, 4), (128, 16)):
        Q, _ = np.linalg.qr(rng.normal(size=(d, k)))
        out.append((f"synthetic_d{d}_k{k}", np.asarray(Q[:, :k], dtype=np.float32)))
    return out


def real_bases(arrays: dict[str, np.ndarray], ranks: tuple[int, ...]) -> list[tuple[str, np.ndarray]]:
    """Реальные INLP-базисы, усечённые до нескольких рангов."""
    out = []
    for key in sorted(k for k in arrays if k.endswith("__W")):
        W = arrays[key]
        if W.ndim != 2 or W.shape[1] == 0:
            continue
        for k in ranks:
            if k <= W.shape[1]:
                out.append((f"{key}[:, :{k}]", np.ascontiguousarray(W[:, :k])))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subspaces", type=Path, default=DEFAULT_SUBSPACES)
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "results" / "steering")
    ap.add_argument("--tag", default="inlp_unit")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    arrays = load_subspaces(args.subspaces)
    if arrays:
        print(f"[1] Подпространства: {args.subspaces.name} ({len(arrays)} массивов)")
    else:
        print(f"[1] {args.subspaces.name} не найден — T1 пропускается, T2–T5 на синтетике")

    bases = synthetic_bases(rng) + real_bases(arrays, ranks=(1, 4))
    print(f"[2] Базисов для T2–T4: {len(bases)}")

    tests = [
        t1_orthonormal(arrays),
        t2_projection_removal(bases, rng),
        t3_center_alpha1(bases, rng),
        t4_alpha_zero(bases, rng),
        t5_rank1_compatibility(rng),
    ]

    print()
    for t in tests:
        if t["passed"] is None:
            print(f"  SKIP  {t['name']}")
            continue
        print(f"  {'PASS' if t['passed'] else 'FAIL':<5} {t['name']}")
        if not t["passed"]:
            for key in ("detail", "detail_numpy", "detail_hook_float32"):
                for row in t.get(key, []):
                    if not row.get("ok", True):
                        print(f"        {row}")

    all_ok = all(t["passed"] for t in tests if t["passed"] is not None)
    out_dir = args.out_root / "inlp_checks" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema": "steering.inlp_unit_checks/v1",
        "tag": args.tag,
        "datetime": datetime.now().isoformat(),
        "subspaces_file": str(args.subspaces.name),
        "subspaces_present": bool(arrays),
        "seed": args.seed,
        "n_bases": len(bases),
        "passed": all_ok,
        "tests": tests,
    }
    out_path = out_dir / "inlp_unit_checks.json"
    out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n=== unit-тесты {'PASSED' if all_ok else 'FAILED'} → {out_path} ===")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
