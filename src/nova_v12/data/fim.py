from __future__ import annotations

import ast
import random
from dataclasses import asdict, dataclass


@dataclass(slots=True)
class FIMRecord:
    id: str
    language: str
    prefix: str
    middle: str
    suffix: str
    source_hash: str
    metadata: dict

    def to_dict(self) -> dict:
        return asdict(self)


def generate_fim_records(
    content: str,
    *,
    language: str,
    source_hash: str,
    count: int = 1,
    seed: int = 42,
    min_middle_chars: int = 8,
    max_middle_fraction: float = 0.75,
) -> list[FIMRecord]:
    spans = _candidate_spans(content, language)
    spans = [
        span
        for span in spans
        if span[1] - span[0] >= min_middle_chars
        and (span[1] - span[0]) <= max(1, int(len(content) * max_middle_fraction))
    ]
    if not spans:
        return []
    rng = random.Random(seed)
    rng.shuffle(spans)
    output: list[FIMRecord] = []
    for index, (start, end) in enumerate(spans[:count]):
        output.append(
            FIMRecord(
                id=f"{source_hash[:16]}-fim-{index}",
                language=language,
                prefix=content[:start],
                middle=content[start:end],
                suffix=content[end:],
                source_hash=source_hash,
                metadata={"start": start, "end": end},
            )
        )
    return output


def format_native_fim(record: FIMRecord, tokens: dict[str, str] | None = None) -> str:
    tokens = tokens or {
        "prefix": "<|fim_prefix|>",
        "suffix": "<|fim_suffix|>",
        "middle": "<|fim_middle|>",
    }
    return (
        f"{tokens['prefix']}{record.prefix}"
        f"{tokens['suffix']}{record.suffix}"
        f"{tokens['middle']}{record.middle}"
    )


def _candidate_spans(content: str, language: str) -> list[tuple[int, int]]:
    if language.lower() == "python":
        try:
            tree = ast.parse(content)
            line_offsets = _line_offsets(content)
            spans: list[tuple[int, int]] = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
                    first, last = node.body[0], node.body[-1]
                    start = line_offsets[first.lineno - 1] + first.col_offset
                    end_line = getattr(last, "end_lineno", last.lineno)
                    end_col = getattr(last, "end_col_offset", 0)
                    end = line_offsets[end_line - 1] + end_col
                    spans.append((start, end))
            if spans:
                return spans
        except SyntaxError:
            pass
    lines = content.splitlines(keepends=True)
    if len(lines) < 4:
        return []
    offsets = _line_offsets(content)
    spans = []
    for start_line in range(1, len(lines) - 1):
        end_line = min(len(lines) - 1, start_line + max(1, len(lines) // 5))
        spans.append((offsets[start_line], offsets[end_line]))
    return spans


def _line_offsets(content: str) -> list[int]:
    offsets = [0]
    for index, char in enumerate(content):
        if char == "\n":
            offsets.append(index + 1)
    return offsets
