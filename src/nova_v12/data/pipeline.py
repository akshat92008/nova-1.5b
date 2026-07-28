from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import yaml

from nova_v12.data.adapters import iter_huggingface_records, iter_jsonl_records
from nova_v12.data.contamination import ContaminationScanner
from nova_v12.data.dedup import MinHashDeduplicator, SQLiteExactDeduplicator
from nova_v12.data.licences import DEFAULT_ALLOWLIST, allowed_licence
from nova_v12.data.quality import score_code
from nova_v12.data.security import scan_sensitive_text
from nova_v12.schemas import CodeRecord


@dataclass(slots=True)
class BuildStats:
    seen: int = 0
    accepted: int = 0
    tokens: int = 0
    rejected: Counter = field(default_factory=Counter)
    languages: Counter = field(default_factory=Counter)
    licences: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seen": self.seen,
            "accepted": self.accepted,
            "tokens": self.tokens,
            "rejected": dict(self.rejected),
            "languages": dict(self.languages),
            "licences": dict(self.licences),
        }


def estimate_tokens(text: str, tokenizer: Any | None = None) -> int:
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False))
    # Explicitly an estimate; manifests record which method was used.
    return max(1, (len(text.encode("utf-8")) + 3) // 4)


def build_data(config_path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    input_cfg = dict(config.get("input", {}))
    output_cfg = dict(config.get("output", {}))
    filters = dict(config.get("filters", {}))
    limits = dict(config.get("limits", {}))

    output_path = Path(output_cfg["records"])
    manifest_path = Path(output_cfg.get("manifest", str(output_path) + ".manifest.json"))
    dedup_path = Path(output_cfg.get("dedup_db", str(output_path) + ".dedup.sqlite3"))
    resume = bool(output_cfg.get("resume", False))
    if output_path.exists() and not resume:
        raise FileExistsError(f"output exists; set output.resume=true to append: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    allowlist = {str(item).lower() for item in filters.get("licence_allowlist", DEFAULT_ALLOWLIST)}
    max_bytes = int(filters.get("max_file_bytes", 1_000_000))
    min_quality = float(filters.get("min_quality_score", 0.70))
    reject_pii = bool(filters.get("reject_pii", True))
    scanner = None
    signature_path = filters.get("contamination_signatures")
    if signature_path and Path(signature_path).exists():
        scanner = ContaminationScanner.from_file(signature_path)
    near = None
    if filters.get("enable_near_dedup", False):
        near = MinHashDeduplicator(float(filters.get("near_dedup_threshold", 0.85)))
    tokenizer = _load_tokenizer(config.get("tokenizer"))

    max_records = int(limits.get("max_records", 0))
    max_tokens = int(limits.get("max_tokens", 0))
    stats = BuildStats()
    mode = "a" if resume else "w"

    with (
        SQLiteExactDeduplicator(dedup_path) as exact,
        output_path.open(mode, encoding="utf-8") as handle,
    ):
        for record in _iter_records(input_cfg):
            stats.seen += 1
            reason = _reject_reason(
                record,
                allowlist=allowlist,
                max_bytes=max_bytes,
                min_quality=min_quality,
                reject_pii=reject_pii,
                scanner=scanner,
                exact=exact,
                near=near,
            )
            if reason:
                stats.rejected[reason] += 1
                continue
            ok, licence = allowed_licence(record.licence, allowlist)
            assert ok
            record.licence = licence
            token_count = estimate_tokens(record.content, tokenizer)
            if max_tokens and stats.tokens + token_count > max_tokens:
                break
            payload = record.to_dict()
            payload["token_count"] = token_count
            payload["quality"] = score_code(
                record.content, record.language, min_score=min_quality
            ).to_dict()
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            stats.accepted += 1
            stats.tokens += token_count
            stats.languages[record.language] += 1
            stats.licences[licence] += 1
            if max_records and stats.accepted >= max_records:
                break

    manifest = {
        "config": config,
        "stats": stats.to_dict(),
        "token_count_method": "tokenizer" if tokenizer is not None else "utf8_bytes_div_4_estimate",
        "output": str(output_path),
        "dedup_db": str(dedup_path),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _iter_records(config: dict[str, Any]) -> Iterator[CodeRecord]:
    kind = str(config.get("kind", "jsonl"))
    adapter = str(config.get("adapter", "auto"))
    if kind == "jsonl":
        paths = config.get("paths", [])
        if not paths:
            raise ValueError("input.paths is required for jsonl input")
        yield from iter_jsonl_records(paths, kind=adapter)
        return
    if kind == "huggingface":
        yield from iter_huggingface_records(
            str(config["dataset_id"]),
            split=str(config.get("split", "train")),
            data_dir=config.get("data_dir"),
            kind=adapter,
            revision=config.get("revision"),
        )
        return
    raise ValueError(f"unsupported input kind: {kind}")


def _reject_reason(
    record: CodeRecord,
    *,
    allowlist: set[str],
    max_bytes: int,
    min_quality: float,
    reject_pii: bool,
    scanner: ContaminationScanner | None,
    exact: SQLiteExactDeduplicator,
    near: MinHashDeduplicator | None,
) -> str:
    if len(record.content.encode("utf-8", errors="replace")) > max_bytes:
        return "too_large"
    allowed, _ = allowed_licence(record.licence, allowlist)
    if not allowed:
        return "licence"
    security = scan_sensitive_text(record.content, flag_pii=reject_pii)
    if security:
        return "sensitive_text"
    quality = score_code(record.content, record.language, min_score=min_quality)
    if not quality.accepted:
        return "quality"
    if scanner and scanner.scan(record.to_dict()):
        return "contamination"
    if not exact.add(record.content, record.source_id):
        return "exact_duplicate"
    if near is not None and not near.add(record.content):
        return "near_duplicate"
    return ""


def _load_tokenizer(config: Any):
    if not config:
        return None
    model_id = config if isinstance(config, str) else config.get("model")
    if not model_id:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for exact token counts") from exc
    return AutoTokenizer.from_pretrained(
        model_id, revision=None if isinstance(config, str) else config.get("revision")
    )
