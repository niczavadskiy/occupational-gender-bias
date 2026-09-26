"""Build polarity-pooled INLP samples from xy_pairs (shared family universe).

Writes:
  steering/samples/inlp_polarity_pool_<set_id>_v1.json          # full + split
  steering/samples/inlp_polarity_pool_<set_id>_train_v1.json
  steering/samples/inlp_polarity_pool_<set_id>_val_v1.json
  steering/samples/inlp_polarity_pool_<set_id>_test_v1.json

Example:
  python -m steering.build_polarity_pool_inlp_sample --set-id promale
  python -m steering.build_polarity_pool_inlp_sample --all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from steering.polarity_pool_inlp import (
    DEFAULT_CONFIG,
    SETS_JSON,
    STEERING_DIR,
    build_pool_sample_doc,
    get_set,
    load_polarity_sets,
    load_yaml,
    sample_paths_for,
    sample_stem_for,
    write_split_sample,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set-id", default=None, help="promale | profemale")
    ap.add_argument("--all", action="store_true", help="build every set in sets JSON")
    ap.add_argument("--sets", type=Path, default=SETS_JSON)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--out-dir", type=Path, default=STEERING_DIR / "samples")
    ap.add_argument("--scale", default=None, help="default: scale field from sets JSON (else 2b)")
    args = ap.parse_args(argv)

    doc_sets = load_polarity_sets(args.sets)
    cfg = load_yaml(args.config)
    scale = args.scale or str(doc_sets.get("scale") or cfg.get("scale") or "2b")
    set_ids = [s["id"] for s in doc_sets["sets"]] if args.all else [args.set_id]
    if not set_ids or set_ids == [None]:
        raise SystemExit("pass --set-id or --all")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for sid in set_ids:
        entry = get_set(doc_sets, sid)
        full = build_pool_sample_doc(entry, cfg, scale=scale)
        # Persist full without nested train/val/test item copies (keep ids only + all items)
        persist = {k: v for k, v in full.items() if not k.startswith("items_")}
        persist["items"] = full["items"]
        paths = sample_paths_for(entry, args.out_dir)
        full_path = paths["full"]
        full_path.write_text(json.dumps(persist, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for split in ("train", "val", "test"):
            write_split_sample(full, split, paths[split])
        print(
            f"[{sid}/{scale}] families={full['n_families']} "
            f"train/val/test={full['split']['n_train']}/{full['split']['n_val']}/{full['split']['n_test']} "
            f"layers={full['probe_peak_layers']} → {sample_stem_for(entry)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
