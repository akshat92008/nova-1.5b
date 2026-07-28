#!/usr/bin/env python3
"""Nova V12 QLoRA SFT on execution-verified atomic patch examples."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from nova_v12.constants import SYSTEM_PROMPT
from nova_v12.dataset import canonical_json, sha256_text, validate_manifest
from nova_v12.schema import ContractError


def to_conversation(example: dict) -> dict:
    """Convert a verified record to TRL conversational prompt-completion format."""
    verification = example.get("verification", {})
    if verification.get("verified") is not True:
        raise ContractError("unverified example reached SFT formatter")
    prompt = example["prompt"]
    completion = canonical_json(example["response"])
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"<|nova_patch|>\n{canonical_json(prompt)}",
            },
        ],
        "completion": [{"role": "assistant", "content": completion}],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Nova V12 verified QLoRA SFT")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-verified", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--full-precision",
        action="store_true",
        help="Disable 4-bit NF4 loading; intended only for sufficiently large GPUs",
    )
    args = parser.parse_args()

    try:
        manifest = validate_manifest(
            args.data_dir,
            minimum_verified=args.minimum_verified,
        )
    except (ContractError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"data gate failed: {exc}\n")
    train_path = args.data_dir / "train.jsonl"
    validation_path = args.data_dir / "validation.jsonl"
    if not train_path.is_file() or not validation_path.is_file():
        parser.exit(2, "data gate failed: train and validation shards are required\n")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        import transformers
        import trl
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        parser.exit(2, f"missing training dependency: {exc}\n")

    if not torch.cuda.is_available():
        parser.exit(2, "CUDA GPU required for the production QLoRA recipe\n")
    use_bf16 = bool(torch.cuda.is_bf16_supported())
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    quantization = None
    if not args.full_precision:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = load_dataset(
        "json",
        data_files={"train": str(train_path), "validation": str(validation_path)},
    )
    columns = raw["train"].column_names
    formatted = raw.map(to_conversation, remove_columns=columns)

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
    )
    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        bf16=use_bf16,
        fp16=not use_bf16,
        tf32=True,
        gradient_checkpointing=True,
        max_length=args.max_length,
        completion_only_loss=True,
        packing=False,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        full_determinism=True,
        model_init_kwargs={
            "revision": args.revision,
            "trust_remote_code": args.trust_remote_code,
            "dtype": compute_dtype,
        },
    )
    trainer = SFTTrainer(
        model=args.base_model,
        args=training_args,
        train_dataset=formatted["train"],
        eval_dataset=formatted["validation"],
        processing_class=tokenizer,
        quantization_config=quantization,
        peft_config=peft_config,
    )
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    run = {
        "schema_version": "nova.sft-run.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "base_revision": args.revision,
        "dataset_manifest_sha256": manifest["manifest_sha256"],
        "verified_examples": manifest["verified_examples"],
        "precision": "bf16" if use_bf16 else "fp16",
        "qlora_nf4": not args.full_precision,
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
    (args.output_dir / "nova_sft_run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
