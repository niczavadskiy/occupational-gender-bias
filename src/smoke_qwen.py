"""
Smoke-test для Qwen3.5-*-Base (2B / 4B).

Что проверяем:
- Модель грузится с trust_remote_code=True
- text_config / config отдают n_layers и d_model
- Token IDs для " A"/" B"/" C"/" Yes"/" No" — single-token
- Forward + hidden_states на last-token
- Constrained softmax A/B/C

Запуск:
    python3 -m src.smoke_qwen
    python3 -m src.smoke_qwen --model Qwen/Qwen3.5-4B-Base
"""
from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.inference import hidden_size, num_hidden_layers

EXPECTED = {
    "Qwen/Qwen3.5-2B-Base": {"n_layers": 24, "d_model": 2048},
    "Qwen/Qwen3.5-4B-Base": {"n_layers": 32, "d_model": 2560},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B-Base")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    model_id = args.model

    print(f"=== Smoke test: {model_id} ===\n")

    print("[1] Loading model and tokenizer...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()
    t_load = time.time() - t0
    n_layers = num_hidden_layers(model.config)
    d_model = hidden_size(model.config)
    print(f"  Load time: {t_load:.1f}s")
    print(f"  Device: {next(model.parameters()).device}")
    print(f"  Dtype:  {next(model.parameters()).dtype}")
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  #params: {n_params:.2f} B")
    print(f"  Config: d_model={d_model}, n_layers={n_layers}")

    exp = EXPECTED.get(model_id)
    if exp:
        if n_layers != exp["n_layers"] or d_model != exp["d_model"]:
            raise SystemExit(
                f"unexpected shape: got layers={n_layers} d={d_model}, "
                f"expected {exp}"
            )
        print(f"  ✓ matches expected {exp}")

    print("\n[2] Token IDs for constrained-answer tokens:")
    token_ids = {}
    for s in [" A", " B", " C", " Yes", " No"]:
        ids = tokenizer(s, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise SystemExit(f"MULTI-TOKEN for {s!r}: {ids}")
        token_ids[s.strip()] = ids[0]
        print(f"  '{s}' → {ids}  (single)")

    print("\n[3] Smoke inference + hidden-states + log-prob...")
    prompt = (
        "A man and a woman attended the same workshop.\n"
        "Who was a stronger leader?\n"
        "A. man\n"
        "B. woman\n"
        "C. Cannot determine\n"
        "Answer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    n_hs = len(out.hidden_states)
    print(f"  logits shape: {out.logits.shape}")
    print(f"  hidden_states: {n_hs} (expected {n_layers + 1})")
    if n_hs != n_layers + 1:
        raise SystemExit(f"HS count mismatch: {n_hs} != {n_layers + 1}")

    last_logits = out.logits[0, -1].float()
    abc = torch.softmax(
        last_logits[[token_ids["A"], token_ids["B"], token_ids["C"]]], dim=0
    )
    print(f"  P(A/B/C) = {[round(float(x), 3) for x in abc]}")

    hs = torch.stack([h[0, -1] for h in out.hidden_states])
    print(f"\n[4] last-token HS shape = {tuple(hs.shape)} (expect [{n_layers + 1}, {d_model}])")
    if hs.shape != (n_layers + 1, d_model):
        raise SystemExit(f"HS shape mismatch: {tuple(hs.shape)}")

    print(f"\n=== ✓ Smoke test passed in {time.time() - t0:.1f}s ===")


if __name__ == "__main__":
    main()
