from __future__ import annotations

import hashlib
import re
import sqlite3
from pathlib import Path


def normalise_for_dedup(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class SQLiteExactDeduplicator:
    """Resumable exact-content deduplication backed by SQLite."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS hashes "
            "(digest TEXT PRIMARY KEY, source_id TEXT, "
            "created_at TEXT DEFAULT CURRENT_TIMESTAMP)"
        )
        self.connection.commit()

    def add(self, text: str, source_id: str = "") -> bool:
        digest = hashlib.sha256(
            normalise_for_dedup(text).encode("utf-8", errors="replace")
        ).hexdigest()
        try:
            self.connection.execute(
                "INSERT INTO hashes(digest, source_id) VALUES (?, ?)", (digest, source_id)
            )
            self.connection.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def __enter__(self) -> "SQLiteExactDeduplicator":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()


class MinHashDeduplicator:
    """Optional approximate deduplication using datasketch MinHashLSH."""

    def __init__(self, threshold: float = 0.85, num_perm: int = 128) -> None:
        try:
            from datasketch import MinHashLSH
        except ImportError as exc:
            raise RuntimeError("install datasketch with the data extra") from exc
        self.threshold = threshold
        self.num_perm = num_perm
        self._lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self._counter = 0

    def _signature(self, text: str):
        from datasketch import MinHash

        tokens = normalise_for_dedup(text).split()
        signature = MinHash(num_perm=self.num_perm)
        if len(tokens) < 5:
            signature.update(" ".join(tokens).encode("utf-8"))
            return signature
        for index in range(len(tokens) - 4):
            signature.update(" ".join(tokens[index : index + 5]).encode("utf-8"))
        return signature

    def add(self, text: str) -> bool:
        signature = self._signature(text)
        if self._lsh.query(signature):
            return False
        key = f"record-{self._counter}"
        self._counter += 1
        self._lsh.insert(key, signature)
        return True
