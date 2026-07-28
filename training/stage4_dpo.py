#!/usr/bin/env python3
"""Nova V12 DPO on execution-ranked preference pairs."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from nova_v12.dataset import canonical_json, sha256_text
from nova_v12.preference import validate_preference_manifest
from nova_v12.schema import ContractError


def main() -> None:
    parser = argparse.ArgumentParser(description="Nova V12 execution-ranked DPO")
    parser.add_argument("--sft-adapter", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-pairs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint")
    args = parser.parse_args()

    try:
        preference = validate_preference_manifest(
            args.data_dir,
            minimum_pairs=args.minimum_pairs,
        )
    except (ContractError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"preference data gate failed: {exc}\n")
    if not (args.sft_adapter / "nova_sft_run.json").is_file():
        parser.exit(2, "SFT adapter is missing nova_sft_run.json provenance\n")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        import transformers
        import trl
        from datasets import load_dataset
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        parser.exit(2, f"missing training dependency: {exc}\n")
    if not torch.cuda.is_available():
        parser.exit(2, "CUDA GPU required for the production DPO recipe\n")
    use_bf16 = bool(torch.cuda.is_bf16_supported())
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.sft_adapter,
        is_trainable=True,
        device_map={"": 0},
    )
    tokenizer = AutoTokenizer.from_pretrained(args.sft_adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset(
        "json",
        data_files=str(args.data_dir / preference["shard"]["path"]),
        split="train",
    )
    config = DPOConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        beta=args.beta,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=not use_bf16,
        tf32=True,
        gradient_checkpointing=True,
        max_length=args.max_length,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        full_determinism=True,
    )
    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    run = {
        "schema_version": "nova.dpo-run.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sft_adapter": str(args.sft_adapter),
        "preference_manifest_sha256": preference["manifest_sha256"],
        "pairs": preference["pairs"],
        "seed": args.seed,
        "metrics": result.metrics,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "trl": trl.__version__,
        },
    }
    run["run_sha256"] = sha256_text(canonical_json(run))
    (args.output_dir / "nova_dpo_run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
