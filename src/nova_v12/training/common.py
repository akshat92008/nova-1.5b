from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

import yaml

from nova_v12.schemas import load_jsonl

MODE_TOKENS = ["<|nova_code|>", "<|nova_edit|>", "<|nova_debug|>", "<|nova_agent|>"]


def load_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError("training config must be a mapping")
    return value


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_records(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for path in paths:
        values.extend(load_jsonl(path))
    return values


def ensure_atomic_tokens(tokenizer: Any, model: Any, tokens: list[str] | None = None) -> list[str]:
    """Add mode tokens only when they are not already atomic tokenizer entries."""
    tokens = tokens or MODE_TOKENS
    missing: list[str] = []
    unk_id = getattr(tokenizer, "unk_token_id", None)
    for token in tokens:
        ids = tokenizer.encode(token, add_special_tokens=False)
        atomic = len(ids) == 1 and (unk_id is None or ids[0] != unk_id)
        if not atomic:
            missing.append(token)
    if missing:
        tokenizer.add_special_tokens({"additional_special_tokens": missing})
        model.resize_token_embeddings(len(tokenizer))
    return missing


def apply_lora(model: Any, config: dict[str, Any], *, train_embeddings: bool = False):
    if not config.get("use_lora", True):
        return model
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError("install the train extra to use LoRA") from exc
    target_modules = config.get("target_modules") or infer_lora_targets(model)
    modules_to_save = ["embed_tokens", "lm_head"] if train_embeddings else None
    lora = LoraConfig(
        r=int(config.get("lora_r", 32)),
        lora_alpha=int(config.get("lora_alpha", 64)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
    )
    return get_peft_model(model, lora)


def infer_lora_targets(model: Any) -> list[str]:
    candidates = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "query_key_value",
        "dense",
        "fc1",
        "fc2",
    }
    present: set[str] = set()
    for name, _ in model.named_modules():
        final = name.rsplit(".", 1)[-1]
        if final in candidates:
            present.add(final)
    if not present:
        raise ValueError("could not infer LoRA target modules; set target_modules in config")
    return sorted(present)


def dtype_from_config(config: dict[str, Any]):
    import torch

    if config.get("bf16", False) and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if config.get("fp16", False) and torch.cuda.is_available():
        return torch.float16
    return torch.float32


def save_run_metadata(
    output_dir: str | Path, config: dict[str, Any], extra: dict[str, Any]
) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    payload = {"config": config, **extra}
    (target / "run_metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
