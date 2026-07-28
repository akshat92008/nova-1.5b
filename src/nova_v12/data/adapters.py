from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from nova_v12.schemas import CodeRecord, load_jsonl


def _licence_text(value: object, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    if isinstance(value, Iterable):
        return "|".join(str(item) for item in value if str(item).strip())
    return str(value)


def adapt_stack_v1(example: dict[str, Any]) -> CodeRecord:
    """Normalise a The Stack v1 record.

    The v1 schema commonly uses `licenses` (list), `repo_name`, `path`,
    `content`, `language` and `hexsha`.
    """
    return CodeRecord(
        source="the-stack-v1",
        repository=str(example.get("repo_name") or example.get("repository_name") or "unknown"),
        revision=str(example.get("hexsha") or example.get("revision") or ""),
        path=str(example.get("path") or ""),
        licence=_licence_text(
            example.get("licenses") or example.get("license") or example.get("licence")
        ),
        language=str(example.get("language") or "unknown").lower(),
        content=str(example.get("content") or ""),
        source_id=str(example.get("max_stars_repo_name") or example.get("hexsha") or ""),
        metadata={
            "original_licenses": example.get("licenses"),
            "size": example.get("size"),
            "ext": example.get("ext"),
        },
    )


def adapt_stack_v2(example: dict[str, Any]) -> CodeRecord:
    """Normalise a The Stack v2 metadata/content record.

    Some v2 streams provide metadata only. The caller must hydrate `content`
    before this adapter is used; missing content fails closed.
    """
    content = example.get("content")
    if not isinstance(content, str) or not content:
        raise ValueError("The Stack v2 record has no hydrated content")
    detected = example.get("detected_licenses") or example.get("licenses")
    return CodeRecord(
        source="the-stack-v2",
        repository=str(example.get("repo_name") or "unknown"),
        revision=str(example.get("revision_id") or example.get("revision") or ""),
        path=str(example.get("path") or ""),
        licence=_licence_text(detected),
        language=str(example.get("language") or "unknown").lower(),
        content=content,
        source_id=str(example.get("blob_id") or example.get("content_id") or ""),
        metadata={
            "detected_licenses": detected,
            "blob_id": example.get("blob_id"),
            "revision_id": example.get("revision_id"),
            "visit_date": example.get("visit_date"),
        },
    )


def adapt_generic(example: dict[str, Any]) -> CodeRecord:
    record = CodeRecord.from_dict(example)
    if not record.path:
        raise ValueError("code record path is required")
    if not record.content:
        raise ValueError("code record content is required")
    return record


def adapt_record(example: dict[str, Any], kind: str = "auto") -> CodeRecord:
    kind = kind.lower()
    if kind == "stack_v1":
        return adapt_stack_v1(example)
    if kind == "stack_v2":
        return adapt_stack_v2(example)
    if kind == "generic":
        return adapt_generic(example)
    if kind != "auto":
        raise ValueError(f"unsupported adapter kind: {kind}")
    if "detected_licenses" in example or "blob_id" in example:
        return adapt_stack_v2(example)
    if "licenses" in example or "hexsha" in example:
        return adapt_stack_v1(example)
    return adapt_generic(example)


def iter_jsonl_records(paths: Iterable[str | Path], *, kind: str = "auto") -> Iterator[CodeRecord]:
    for path in paths:
        for value in load_jsonl(path):
            yield adapt_record(value, kind=kind)


def iter_huggingface_records(
    dataset_id: str,
    *,
    split: str = "train",
    data_dir: str | None = None,
    kind: str = "auto",
    revision: str | None = None,
) -> Iterator[CodeRecord]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install the data extra: pip install -e '.[data]'") from exc
    kwargs: dict[str, Any] = {"split": split, "streaming": True}
    if data_dir:
        kwargs["data_dir"] = data_dir
    if revision:
        kwargs["revision"] = revision
    dataset = load_dataset(dataset_id, **kwargs)
    for example in dataset:
        try:
            yield adapt_record(dict(example), kind=kind)
        except ValueError:
            continue
