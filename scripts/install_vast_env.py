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
  PIN_VAST_ENV=0              skip automatic install (still runs when --force)
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


def run(cmd: list[str], *, check: bool = True, env: dict[str, str] | None = None) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check, env=env).returncode


def _pip_env() -> dict[str, str]:
    """Drop hash-enforcement env that breaks PyTorch CDN / truncated wheels."""
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("PIP_") and "HASH" in key.upper():
            env.pop(key, None)
    env.pop("PIP_REQUIRE_HASHES", None)
    # Avoid writing huge wheels to a full user cache mid-download.
    env.setdefault("PIP_NO_CACHE_DIR", "1")
    return env


def pip(*args: str, check: bool = True) -> int:
    return run([sys.executable, "-m", "pip", *args], check=check, env=_pip_env())


def _free_gb(path: str) -> float:
    try:
        st = os.statvfs(path)
        return (st.f_bavail * st.f_frsize) / (1024**3)
    except OSError:
        return -1.0


def _assert_disk_for_torch(min_gb: float = 6.0) -> None:
    candidates = ["/workspace", "/tmp", "/opt/conda", "/"]
    best = max((_free_gb(p), p) for p in candidates)
    free, where = best
    print(f"  disk free: {free:.1f} GiB on {where} (need ≥{min_gb})", flush=True)
    if free >= 0 and free < min_gb:
        raise SystemExit(
            f"Not enough disk for torch≈0.8GiB wheel (free {free:.1f} GiB on {where}).\n"
            f"  Free space (old wheels/tars/HF cache), then re-run:\n"
            f"    df -h\n"
            f"    rm -rf /tmp/pip-* /root/.cache/pip /workspace/*.tar.gz\n"
            f"    {sys.executable} scripts/install_vast_env.py --force"
        )


def _curl_download(url: str, dest: Path, *, min_bytes: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        dest.unlink()
    cmd = [
        "curl",
        "-fL",
        "--retry",
        "5",
        "--retry-all-errors",
        "--retry-delay",
        "2",
        "--connect-timeout",
        "30",
        "-o",
        str(dest),
        url,
    ]
    run(cmd, check=True)
    size = dest.stat().st_size
    if size < min_bytes:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"download too small ({size} < {min_bytes}): {url}")


def _install_torch_wheels_via_curl(lock: dict) -> None:
    """Bypass pip's streaming download (URL #sha256 fails on truncated CDN reads)."""
    import tempfile

    tag = lock["cuda"]["tag"]
    torch_v = lock["pytorch"]["torch"]
    tv_v = lock["pytorch"]["torchvision"]
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    # Official cu121 manylinux wheels used by Vast (cp310).
    torch_name = f"torch-{torch_v}%2B{tag}-{py_tag}-{py_tag}-linux_x86_64.whl"
    tv_name = f"torchvision-{tv_v}%2B{tag}-{py_tag}-{py_tag}-linux_x86_64.whl"
    base = f"https://download.pytorch.org/whl/{tag}"
    torch_url = f"{base}/{torch_name}"
    tv_url = f"{base}/{tv_name}"

    with tempfile.TemporaryDirectory(prefix="vast_torch_") as td:
        tdir = Path(td)
        torch_whl = tdir / torch_name.replace("%2B", "+")
        tv_whl = tdir / tv_name.replace("%2B", "+")
        print(f"  curl torch wheel → {torch_whl.name}", flush=True)
        _curl_download(torch_url, torch_whl, min_bytes=500_000_000)
        print(f"  curl torchvision wheel → {tv_whl.name}", flush=True)
        _curl_download(tv_url, tv_whl, min_bytes=1_000_000)
        # Replace broken importable-but-cudnn-missing install in one shot.
        pip("uninstall", "-y", "torch", "torchvision", check=False)
        pip(
            "install",
            "--no-cache-dir",
            "--force-reinstall",
            str(torch_whl),
            str(tv_whl),
        )


def install_torch_stack(lock: dict) -> None:
    tag = lock["cuda"]["tag"]
    idx = lock["cuda"]["torch_index_url"]
    torch_v = lock["pytorch"]["torch"]
    tv_v = lock["pytorch"]["torchvision"]
    force = bool(lock.get("pytorch", {}).get("force_reinstall", True))

    _assert_disk_for_torch(6.0)

    # Strip audio first so a broken wheel cannot linger beside the new torch.
    for pkg in lock["pytorch"].get("uninstall") or []:
        pip("uninstall", "-y", pkg, check=False)

    # Drop stale pip HTTP cache (partial torch wheels → hash mismatch).
    pip("cache", "purge", check=False)

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

    last_err: BaseException | None = None
    for attempt in range(1, 4):
        try:
            print(f"  pip torch install attempt {attempt}/3…", flush=True)
            pip(*args)
            print(f"  installed torch=={torch_v} torchvision=={tv_v} ({tag}) force={force}")
            return
        except subprocess.CalledProcessError as exc:
            last_err = exc
            print(
                f"  WARN: pip torch install failed (attempt {attempt}/3) — "
                f"often truncated CDN download / hash mismatch",
                flush=True,
            )
            if attempt >= 2:
                try:
                    print("  fallback: curl full wheels then local pip install…", flush=True)
                    _install_torch_wheels_via_curl(lock)
                    print(f"  installed torch=={torch_v} torchvision=={tv_v} ({tag}) via curl")
                    return
                except Exception as curl_exc:  # noqa: BLE001
                    last_err = curl_exc
                    print(f"  WARN: curl fallback failed: {curl_exc}", flush=True)
    assert last_err is not None
    raise SystemExit(f"torch install failed after retries: {last_err}") from last_err



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


def _installed_pin(dist_name: str) -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover
        from importlib_metadata import PackageNotFoundError, version  # type: ignore
    try:
        return f"{dist_name}=={version(dist_name)}"
    except PackageNotFoundError:
        return None


def install_pip_lock() -> None:
    """Install non-torch pins. Must not upgrade PyPI torch (2.14+cu13).

    ``pip install -U -r …`` lets accelerate≥0.33 pull the latest torch from
    PyPI (CUDA 13 wheels, hash/cache failures, disk blow-up). Torch is installed
    separately from the cu121 index in ``install_torch_stack``.
    """
    if not REQ_LOCK.is_file():
        raise SystemExit(f"missing {REQ_LOCK}")
    import tempfile

    # Pin currently installed torch stack so accelerate cannot resolve 2.14+cu13.
    pins = [
        p
        for p in (
            _installed_pin("torch"),
            _installed_pin("torchvision"),
            _installed_pin("torchaudio"),
            _installed_pin("triton"),
        )
        if p
    ]
    # If torch is missing, install_torch_stack should have run first; still refuse
    # open-ended torch from PyPI by requiring a high lower-bound never used as target.
    if not any(p.startswith("torch==") for p in pins):
        pins.append("torch==2.5.1")
    with tempfile.NamedTemporaryFile(
        "w", suffix="-torch-constraint.txt", delete=False, encoding="utf-8"
    ) as fh:
        fh.write("\n".join(pins) + "\n")
        constraint_path = fh.name
    try:
        print(f"  pip lock constraints: {pins}", flush=True)
        pip(
            "install",
            "--no-cache-dir",
            "--upgrade-strategy",
            "only-if-needed",
            "-c",
            constraint_path,
            "-r",
            str(REQ_LOCK),
        )
    finally:
        try:
            os.unlink(constraint_path)
        except OSError:
            pass


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

    # PIN_VAST_ENV=0 skips the *automatic* install path. Explicit --force must
    # always repair (Ministral scripts hit this when libcudnn/torch is broken).
    if os.environ.get("PIN_VAST_ENV", "1") == "0" and not args.force and not args.check_only:
        print("PIN_VAST_ENV=0 — skip (pass --force to repair anyway)")
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
        # Torch first so accelerate/transformers see an existing CUDA build and
        # do not resolve bare PyPI torch==2.14+cu13.
        install_torch_stack(lock)
        install_pip_lock()
        if lock.get("optional", {}).get("flash-linear-attention", {}).get("install") and os.environ.get(
            "SKIP_FLA", "0"
        ) != "1":
            # --no-deps: FLA 0.5.x depends on torch 2.14+cu13 and would overwrite cu121 lock.
            pip("install", "-U", "--no-deps", "flash-linear-attention", check=False)

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
