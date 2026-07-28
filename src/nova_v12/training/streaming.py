from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

from nova_v12.data.fim import format_native_fim, generate_fim_records


def _record_text(record: dict[str, Any], fim_rate: float, seed: int) -> str:
    if record.get("formatted"):
        return str(record["formatted"])
    content = str(record.get("content", ""))
    if not content:
        return ""
    stable = int(
        hashlib.sha256((str(record.get("content_hash", "")) + str(seed)).encode()).hexdigest()[:16],
        16,
    )
    rng = random.Random(stable)
    if rng.random() < fim_rate:
        records = generate_fim_records(
            content,
            language=str(record.get("language", "text")),
            source_hash=str(
                record.get("content_hash", hashlib.sha256(content.encode()).hexdigest())
            ),
            count=1,
            seed=stable,
        )
        if records:
            return format_native_fim(records[0])
    return content


class PackedJSONLIterableDataset:
    """Worker-aware, deterministic packed causal-LM dataset.

    Records are streamed from JSONL and packed into fixed-length sequences. A
    model run using this dataset must use `max_steps`, because an iterable
    dataset has no reliable length.
    """

    def __init__(
        self,
        paths: Iterable[str | Path],
        tokenizer: Any,
        *,
        sequence_length: int,
        fim_rate: float = 0.35,
        seed: int = 42,
        repeat: bool = True,
    ) -> None:
        try:
            from torch.utils.data import IterableDataset
        except ImportError as exc:
            raise RuntimeError("torch is required for streaming training") from exc
        self._base_class = IterableDataset
        self.paths = [Path(path) for path in paths]
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.fim_rate = fim_rate
        self.seed = seed
        self.repeat = repeat

    def __iter__(self):
        import torch
        from torch.utils.data import get_worker_info

        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        workers = worker.num_workers if worker else 1
        eos = self.tokenizer.eos_token_id
        if eos is None:
            raise ValueError("tokenizer requires eos_token_id")
        epoch = 0
        while True:
            buffer: list[int] = []
            global_index = 0
            for path in self.paths:
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if global_index % workers != worker_id:
                            global_index += 1
                            continue
                        global_index += 1
                        stripped = line.strip()
                        if not stripped:
                            continue
                        record = json.loads(stripped)
                        text = _record_text(record, self.fim_rate, self.seed + epoch)
                        if not text:
                            continue
                        buffer.extend(self.tokenizer.encode(text, add_special_tokens=False))
                        buffer.append(eos)
                        while len(buffer) >= self.sequence_length:
                            ids = buffer[: self.sequence_length]
                            del buffer[: self.sequence_length]
                            tensor = torch.tensor(ids, dtype=torch.long)
                            yield {
                                "input_ids": tensor,
                                "attention_mask": torch.ones_like(tensor),
                                "labels": tensor.clone(),
                            }
            if not self.repeat:
                break
            epoch += 1

    # Trainer checks isinstance(dataset, torch IterableDataset). Dynamic
    # inheritance is not possible after construction, so expose the protocol
    # and wrap through `as_torch_dataset` below.


def as_torch_dataset(dataset: PackedJSONLIterableDataset):
    from torch.utils.data import IterableDataset

    class Wrapper(IterableDataset):
        def __iter__(self):
            yield from iter(dataset)

    return Wrapper()
