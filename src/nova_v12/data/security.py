from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(slots=True)
class SecurityFinding:
    kind: str
    line: int
    excerpt: str


_PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    "generic_secret": re.compile(
        r"(?i)\b(?:api[_-]?key|secret|token|password)\s*[:=]\s*['\"]([A-Za-z0-9_\-/.+=]{16,})['\"]"
    ),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
}

_EMAIL = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w.-])")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d ()-]{8,}\d)(?!\d)")


def scan_sensitive_text(text: str, *, flag_pii: bool = True) -> list[SecurityFinding]:
    findings: list[SecurityFinding] = []
    lines = text.splitlines()
    for index, line in enumerate(lines, 1):
        for kind, pattern in _PATTERNS.items():
            if pattern.search(line):
                findings.append(SecurityFinding(kind, index, line[:200]))
        if flag_pii and _EMAIL.search(line) and not _looks_like_example_email(line):
            findings.append(SecurityFinding("email", index, line[:200]))
        if flag_pii and _PHONE.search(line) and not re.search(r"\b(?:555|000)[- )]", line):
            findings.append(SecurityFinding("phone", index, line[:200]))
    return findings


def _looks_like_example_email(line: str) -> bool:
    lowered = line.lower()
    return any(
        domain in lowered for domain in ("example.com", "example.org", "test.com", "localhost")
    )
