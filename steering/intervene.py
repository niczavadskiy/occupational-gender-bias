"""
Хук интервенции в residual stream + constrained A/B скоринг.

Точка входа для Stage A/B/C и для capability-прогонов на MMLU-Pro: модель
грузится один раз, кандидаты применяются по очереди через
`with steered(model, spec): ...`.

Индексация слоя совпадает с hidden_states.npz исходного прогона:
    0        = выход эмбеддингов
    1..24    = выход блока L (то есть decoder_layers[L-1])
Модифицируется ТОЛЬКО последняя позиция промпта — та же точка, где снимался HS
для проб. При одиночном forward без генерации это позиция, из которой читаются
логиты ответа.

Численность: перебивка считается в float32, но записывается в dtype модели.
Возмущение мало относительно ||h||≈12 (|Δ| на компоненту ~1e-3), поэтому bf16
съедает заметную часть шага — для боевых прогонов нужен float32.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

CONSTRAINED_TOKENS = (" A", " B", " C")


@dataclass(frozen=True)
class InterventionSpec:
    """Одна интервенция на одном слое (кандидат может задавать несколько)."""

    layer: int
    w: np.ndarray
    kind: str  # center | project_out | shift
    alpha: float | None = None
    beta: float | None = None
    c: float = 0.0
    sigma: float = 1.0

    def expected_s_after(self, s_before: np.ndarray) -> np.ndarray:
        if self.kind == "center":
            return s_before - float(self.alpha) * (s_before - self.c)
        if self.kind == "project_out":
            return np.zeros_like(s_before)
        if self.kind == "shift":
            return s_before + float(self.beta) * self.sigma
        raise ValueError(f"unknown intervention kind {self.kind!r}")


class ProjectionTrace:
    """Проекции s до/после по каждому вызову хука (диагностика корректности)."""

    def __init__(self) -> None:
        self.before: dict[int, float] = {}
        self.after: dict[int, float] = {}

    def clear(self) -> None:
        self.before.clear()
        self.after.clear()


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    """ModuleList блоков декодера (обходит text-обёртки мультимодальных конфигов)."""
    n_layers = num_hidden_layers(model.config)
    found = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.ModuleList) and len(module) == n_layers and name.endswith("layers")
    ]
    if not found:
        raise RuntimeError(f"не найден ModuleList из {n_layers} блоков декодера")
    return min(found, key=lambda pair: len(pair[0]))[1]


def embedding_module(model: nn.Module) -> nn.Module:
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding) and name.endswith("embed_tokens"):
            return module
    raise RuntimeError("не найден embed_tokens")


def num_hidden_layers(config) -> int:
    for cfg in (getattr(config, "text_config", None), config):
        if cfg is not None and getattr(cfg, "num_hidden_layers", None):
            return int(cfg.num_hidden_layers)
    raise RuntimeError("не удалось определить num_hidden_layers")


def hidden_size(config) -> int:
    for cfg in (getattr(config, "text_config", None), config):
        if cfg is not None and getattr(cfg, "hidden_size", None):
            return int(cfg.hidden_size)
    raise RuntimeError("не удалось определить hidden_size")


def _apply(h_last: torch.Tensor, spec: InterventionSpec, w: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    h32 = h_last.to(torch.float32)
    s = float(torch.dot(h32, w))
    if spec.kind == "center":
        delta = -float(spec.alpha) * (s - spec.c)
    elif spec.kind == "project_out":
        delta = -s
    elif spec.kind == "shift":
        delta = float(spec.beta) * spec.sigma
    else:
        raise ValueError(f"unknown intervention kind {spec.kind!r}")
    new_h = (h32 + delta * w).to(h_last.dtype)
    s_after = float(torch.dot(new_h.to(torch.float32), w))
    return new_h, s, s_after


@contextlib.contextmanager
def steered(
    model: nn.Module,
    specs: list[InterventionSpec],
    trace: ProjectionTrace | None = None,
) -> Iterator[None]:
    """Вешает forward-хуки на указанные слои на время блока."""
    if not specs:
        yield
        return

    layers = decoder_layers(model)
    handles = []
    device = next(model.parameters()).device

    def make_hook(spec: InterventionSpec):
        w = torch.as_tensor(spec.w, dtype=torch.float32, device=device)

        def hook(_module, _args, output):
            is_tuple = isinstance(output, tuple)
            hidden = output[0] if is_tuple else output
            new_h, s_before, s_after = _apply(hidden[0, -1, :], spec, w)
            hidden = hidden.clone()
            hidden[0, -1, :] = new_h
            if trace is not None:
                trace.before[spec.layer] = s_before
                trace.after[spec.layer] = s_after
            return (hidden, *output[1:]) if is_tuple else hidden

        return hook

    try:
        for spec in specs:
            if spec.layer == 0:
                target = embedding_module(model)
            else:
                target = layers[spec.layer - 1]
            handles.append(target.register_forward_hook(make_hook(spec)))
        yield
    finally:
        for h in handles:
            h.remove()


class Scorer:
    """Constrained A/B(/C) скоринг — точная копия логики src/inference.py."""

    def __init__(self, model: nn.Module, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.tok_ids = {
            label.strip(): tokenizer(label, add_special_tokens=False).input_ids[0]
            for label in CONSTRAINED_TOKENS
        }
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def score(self, prompt: str, valid_labels: list[str]) -> dict:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        last_logits = out.logits[0, -1].float()
        logits = {k: float(last_logits[v]) for k, v in self.tok_ids.items()}
        log_probs = F.log_softmax(last_logits, dim=0)
        valid = torch.tensor([logits[l] for l in valid_labels])
        probs = F.softmax(valid, dim=0).tolist()
        row = {
            "n_prompt_tokens": int(inputs["input_ids"].shape[1]),
            "choice": valid_labels[int(np.argmax(probs))],
            "valid_labels": list(valid_labels),
        }
        for label, tok in self.tok_ids.items():
            row[f"logit_{label}"] = logits[label]
            row[f"logprob_vocab_{label}"] = float(log_probs[tok])
        for label, p in zip(valid_labels, probs, strict=True):
            row[f"prob_constrained_{label}"] = float(p)
        return row


def load_model(model_path: str | Path, *, dtype: str = "float32", device: str = "cpu"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def specs_for_candidate(
    candidate: dict,
    vectors: dict[str, np.ndarray],
    calib: dict[str, dict],
) -> list[InterventionSpec]:
    """Кандидат из h1_candidates_*.json → список InterventionSpec."""
    seed = candidate.get("random_seed")
    out = []
    for layer in candidate["layers"]:
        key = f"{candidate['vector_id']}__L{layer}"
        if seed is not None:
            key = f"{key}__s{seed}"
        if key not in vectors:
            raise KeyError(f"нет вектора {key} в h1_vectors")
        meta = calib[key]
        kind = candidate["intervention"]
        out.append(
            InterventionSpec(
                layer=layer,
                w=np.asarray(vectors[key], dtype=np.float32),
                kind=kind,
                alpha=candidate.get("alpha"),
                beta=candidate.get("beta"),
                c=float(meta["c"]) if kind == "center" else 0.0,
                sigma=float(meta["sigma_train"]),
            )
        )
    return out
