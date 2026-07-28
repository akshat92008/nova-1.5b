from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(slots=True)
class ContaminationFinding:
    benchmark: str
    signature_id: str
    field: str
    method: str
    excerpt: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def normalise_text(text: str) -> str:
    text = text.lower().replace("\r\n", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class ContaminationScanner:
    def __init__(self, signatures: Iterable[dict[str, Any]]) -> None:
        self.signatures: list[dict[str, str]] = []
        for item in signatures:
            text = normalise_text(str(item.get("text", "")))
            if not text:
                continue
            self.signatures.append(
                {
                    "benchmark": str(item.get("benchmark", "unknown")),
                    "id": str(item.get("id", hashlib.sha256(text.encode()).hexdigest()[:16])),
                    "text": text,
                    "hash": hashlib.sha256(text.encode()).hexdigest(),
                }
            )

    @classmethod
    def from_file(cls, path: str | Path) -> "ContaminationScanner":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        signatures = payload.get("signatures", payload) if isinstance(payload, dict) else payload
        if not isinstance(signatures, list):
            raise ValueError("contamination signature file must contain a list")
        return cls(signatures)

    def scan(self, value: Any) -> list[ContaminationFinding]:
        findings: list[ContaminationFinding] = []
        for field, text in iter_text_fields(value):
            normalised = normalise_text(text)
            digest = hashlib.sha256(normalised.encode()).hexdigest()
            for signature in self.signatures:
                method = ""
                if digest == signature["hash"]:
                    method = "exact_hash"
                elif len(signature["text"]) >= 12 and signature["text"] in normalised:
                    method = "substring"
                if method:
                    position = normalised.find(signature["text"])
                    excerpt = normalised[
                        max(0, position - 80) : position + len(signature["text"]) + 80
                    ]
                    findings.append(
                        ContaminationFinding(
                            signature["benchmark"], signature["id"], field, method, excerpt
                        )
                    )
        return findings


def iter_text_fields(value: Any, prefix: str = "$"):
    if isinstance(value, str):
        yield prefix, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from iter_text_fields(item, f"{prefix}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from iter_text_fields(item, f"{prefix}[{index}]")
