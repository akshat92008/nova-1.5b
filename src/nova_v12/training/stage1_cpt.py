from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from nova_v12.data.fim import format_native_fim, generate_fim_records
from nova_v12.training.common import (
    apply_lora,
    dtype_from_config,
    ensure_atomic_tokens,
    load_config,
    load_records,
    save_run_metadata,
    set_seed,
)


def _text_for_record(record: dict[str, Any], *, fim_rate: float, rng: random.Random) -> str:
    if record.get("formatted"):
        return str(record["formatted"])
    content = str(record.get("content", ""))
    if not content:
        raise ValueError("CPT record requires content or formatted")
    if rng.random() < fim_rate:
        generated = generate_fim_records(
            content,
            language=str(record.get("language", "text")),
            source_hash=str(record.get("content_hash", "unknown")),
            count=1,
            seed=rng.randint(0, 2**31 - 1),
        )
        if generated:
            return format_native_fim(generated[0])
    return content


def train(config_path: str | Path) -> None:
    try:
        from datasets import Dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
            default_data_collator,
        )
    except ImportError as exc:
        raise RuntimeError("install the train extra: pip install -e '.[train]'") from exc

    config = load_config(config_path)
    seed = int(config.get("seed", 42))
    set_seed(seed)
    base_model = str(config["base_model"])
    revision = config.get("revision")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        revision=revision,
        trust_remote_code=bool(config.get("trust_remote_code", False)),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        revision=revision,
        trust_remote_code=bool(config.get("trust_remote_code", False)),
        torch_dtype=dtype_from_config(config),
        device_map="auto" if config.get("device_map", "auto") == "auto" else None,
    )
    added = ensure_atomic_tokens(tokenizer, model)
    model = apply_lora(model, config, train_embeddings=bool(added))

    rng = random.Random(seed)
    fim_rate = float(config.get("fim_rate", 0.35))
    max_length = int(config.get("max_length", 4096))
    streaming = bool(config.get("streaming", False))
    validation_records = load_records(config.get("validation_files", []))
    validation_texts = [
        _text_for_record(item, fim_rate=fim_rate, rng=rng) for item in validation_records
    ]

    def tokenise(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"], truncation=True, max_length=max_length, add_special_tokens=True
        )

    if streaming:
        from nova_v12.training.streaming import PackedJSONLIterableDataset, as_torch_dataset

        if int(config.get("max_steps", 0)) <= 0:
            raise ValueError("streaming CPT requires a positive max_steps")
        packed = PackedJSONLIterableDataset(
            config.get("train_files", []),
            tokenizer,
            sequence_length=max_length,
            fim_rate=fim_rate,
            seed=seed,
            repeat=True,
        )
        train_dataset = as_torch_dataset(packed)
        train_record_count = None
        data_collator = default_data_collator
    else:
        train_records = load_records(config.get("train_files", []))
        train_texts = [_text_for_record(item, fim_rate=fim_rate, rng=rng) for item in train_records]
        train_dataset = Dataset.from_dict({"text": train_texts}).map(
            tokenise, batched=True, remove_columns=["text"]
        )
        train_record_count = len(train_texts)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    eval_dataset = Dataset.from_dict({"text": validation_texts}).map(
        tokenise, batched=True, remove_columns=["text"]
    )
    arguments = TrainingArguments(
        output_dir=str(config["output_dir"]),
        learning_rate=float(config.get("learning_rate", 1e-4)),
        num_train_epochs=float(config.get("num_train_epochs", 1)),
        max_steps=int(config.get("max_steps", -1)),
        per_device_train_batch_size=int(config.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(config.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 16)),
        logging_steps=int(config.get("logging_steps", 10)),
        save_steps=int(config.get("save_steps", 250)),
        eval_strategy="steps" if validation_texts else "no",
        eval_steps=int(config.get("eval_steps", config.get("save_steps", 250))),
        warmup_ratio=float(config.get("warmup_ratio", 0.03)),
        bf16=bool(config.get("bf16", False)),
        fp16=bool(config.get("fp16", False)),
        report_to=config.get("report_to", "none"),
        seed=seed,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if validation_texts else None,
        data_collator=data_collator,
    )
    trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint"))
    trainer.save_model(str(config["output_dir"]))
    tokenizer.save_pretrained(str(config["output_dir"]))
    save_run_metadata(
        config["output_dir"],
        config,
        {
            "added_tokens": added,
            "train_records": train_record_count,
            "validation_records": len(validation_texts),
            "streaming": streaming,
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    train(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
