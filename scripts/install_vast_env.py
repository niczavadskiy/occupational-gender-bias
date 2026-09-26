#!/usr/bin/env python3
"""Install / verify the frozen Vast steering environment from git lock.

Lock file (default):
  scripts/env/vast_steering_cu121.json

Usage on Vast after git pull:

  cd /workspace/occupational-gender-bias
  python scripts/install_vast_env.py
  python scripts/install_vast_env.py --check-only
  python scripts/install_vast_env.py --force   # always reinstall torch stack

Env:
  PIN_VAST_ENV=0              skip (no-op exit 0)
  SKIP_FLA=1                 do not try flash-linear-attention
  FORCE_TORCH_REINSTALL=1    force torch/torchvision even if versions match (default 1)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_LOCK = HERE / "env" / "vast_steering_cu121.json"
REQ_LOCK = HERE / "requirements-vast-lock.txt"


def run(cmd: list[str], *, check: bool = True) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check).returncode


def pip(*args: str, check: bool = True) -> int:
    return run([sys.executable, "-m", "pip", *args], check=check)


def load_lock(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ver(mod: str) -> str | None:
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "?")
    except Exception:
        return None


def current_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {
        "numpy": _ver("numpy"),
        "torch": None,
        "torchvision": None,
        "transformers": _ver("transformers"),
        "cuda_available": None,
        "torch_cuda": None,
        "torch_file": None,
    }
    try:
        import torch

        out["torch"] = torch.__version__
        out["cuda_available"] = str(torch.cuda.is_available())
        out["torch_cuda"] = str(torch.version.cuda)
        out["torch_file"] = getattr(torch, "__file__", "?")
    except Exception as e:
        out["torch"] = f"IMPORT_FAIL:{type(e).__name__}"
    try:
        import torchvision

        out["torchvision"] = torchvision.__version__
    except Exception:
        out["torchvision"] = None
    return out


def torch_integrity_ok() -> tuple[bool, str]:
    """Cold-import check in a subprocess (catches broken conda+pip torch)._C)."""
    code = r"""
import sys
try:
    import torch
    # Broken hybrid installs often fail here on Tensor metaclass:
    _ = torch._C
    if not hasattr(torch._C, "_dlpack_exchange_api"):
        # older ok builds may lack this; still require a tensor op
        pass
    x = torch.zeros(1)
    _ = float(x.item())
    if not torch.cuda.is_available():
        print("NO_CUDA")
        sys.exit(3)
    # Touch CUDA once — catches ABI mismatches that CPU import misses
    y = torch.zeros(1, device="cuda")
    _ = float(y.item())
    print(f"OK {torch.__version__} {torch.__file__}")
    sys.exit(0)
except Exception as e:
    print(f"FAIL {type(e).__name__}: {e}")
    sys.exit(2)
"""
    p = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    msg = (p.stdout or p.stderr or "").strip()
    return p.returncode == 0, msg


def matches(lock: dict, cur: dict[str, str | None]) -> bool:
    want_torch = lock["pytorch"]["torch"]
    want_tv = lock["pytorch"]["torchvision"]
    want_tr = lock["pip"]["transformers"]
    want_np = lock["pip"]["numpy"]
    t = cur.get("torch") or ""
    tv = cur.get("torchvision") or ""
    if t.startswith("IMPORT_FAIL"):
        return False
    if want_torch not in t:
        return False
    if want_tv not in tv:
        return False
    if (cur.get("transformers") or "") != want_tr:
        return False
    if (cur.get("numpy") or "") != want_np:
        return False
    if lock["sanity"].get("require_cuda") and cur.get("cuda_available") != "True":
        return False
    return True


def install_torch_stack(lock: dict) -> None:
    tag = lock["cuda"]["tag"]
    idx = lock["cuda"]["torch_index_url"]
    torch_v = lock["pytorch"]["torch"]
    tv_v = lock["pytorch"]["torchvision"]
    force = bool(lock.get("pytorch", {}).get("force_reinstall", True))

    # Strip audio first so a broken wheel cannot linger beside the new torch.
    for pkg in lock["pytorch"].get("uninstall") or []:
        pip("uninstall", "-y", pkg, check=False)
    pip("uninstall", "-y", "torch", "torchvision", check=False)

    args = ["install", "--no-cache-dir"]
    if force:
        # Critical on Vast conda images: version strings can look right while
        # torch._C is a leftover from a previous/hybrid install
        # (AttributeError: _dlpack_exchange_api).
        args.append("--force-reinstall")
    args += [
        f"torch=={torch_v}",
        f"torchvision=={tv_v}",
        "--index-url",
        idx,
    ]
    pip(*args)
    print(f"  installed torch=={torch_v} torchvision=={tv_v} ({tag}) force={force}")


def install_pip_lock() -> None:
    if not REQ_LOCK.is_file():
        raise SystemExit(f"missing {REQ_LOCK}")
    pip("install", "-U", "-r", str(REQ_LOCK))


def sanity(lock: dict) -> None:
    import importlib

    ok, msg = torch_integrity_ok()
    if not ok:
        raise SystemExit(
            f"torch integrity failed (cold import):\n  {msg}\n"
            f"  Fix: {sys.executable} scripts/install_vast_env.py --force"
        )
    print(f"  OK torch cold-import: {msg}")

    import numpy as np
    import torch
    import transformers

    if int(np.__version__.split(".")[0]) >= 2:
        raise SystemExit(f"numpy {np.__version__} >= 2 — need numpy<2")
    for mod in lock["sanity"]["import_modules"]:
        importlib.import_module(mod)
        print(f"  OK import {mod}")
    print(
        f"  OK numpy={np.__version__} torch={torch.__version__} "
        f"cuda={torch.cuda.is_available()} transformers={transformers.__version__}"
    )
    try:
        import torchvision

        print(f"  OK torchvision={torchvision.__version__}")
    except Exception as e:
        raise SystemExit(f"torchvision import failed: {e}") from e


def _env_force_torch() -> bool:
    return os.environ.get("FORCE_TORCH_REINSTALL", "1") != "0"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument(
        "--force",
        action="store_true",
        help="reinstall torch stack even if versions match",
    )
    args = ap.parse_args(argv)

    if os.environ.get("PIN_VAST_ENV", "1") == "0":
        print("PIN_VAST_ENV=0 — skip")
        return 0

    lock = load_lock(args.lock)
    print(f"lock: {args.lock} ({lock['name']})")
    print(f"python: {sys.executable}")
    cur = current_versions()
    print("current:", cur)
    ok_int, int_msg = torch_integrity_ok()
    print(f"torch integrity: {'OK' if ok_int else 'FAIL'} ({int_msg})")

    if args.check_only:
        ok = matches(lock, cur) and ok_int
        print("MATCH" if ok else "MISMATCH")
        if ok:
            sanity(lock)
            return 0
        return 2

    force = args.force or _env_force_torch() or not ok_int
    if matches(lock, cur) and ok_int and not force:
        print("versions+integrity match lock — skip reinstall")
    else:
        reason = []
        if not matches(lock, cur):
            reason.append("version mismatch")
        if not ok_int:
            reason.append("torch integrity fail")
        if force:
            reason.append("force reinstall")
        print(f"installing pinned stack ({', '.join(reason) or 'requested'})…")
        install_pip_lock()
        install_torch_stack(lock)
        if lock.get("optional", {}).get("flash-linear-attention", {}).get("install") and os.environ.get(
            "SKIP_FLA", "0"
        ) != "1":
            pip("install", "-U", "flash-linear-attention", check=False)

    # Always strip torchaudio for text-only path
    for pkg in lock["pytorch"].get("uninstall") or []:
        pip("uninstall", "-y", pkg, check=False)

    # Re-check in a fresh subprocess after uninstalls
    ok_int, int_msg = torch_integrity_ok()
    if not ok_int:
        print(f"torch broken after install ({int_msg}) — force-reinstall once more…")
        install_torch_stack(lock)
        for pkg in lock["pytorch"].get("uninstall") or []:
            pip("uninstall", "-y", pkg, check=False)

    sanity(lock)
    print("vast env OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
