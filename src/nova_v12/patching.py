from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class PatchApplication:
    ok: bool
    changed_files: list[str]
    error: str = ""


def safe_resolve(root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise ValueError(f"path must be non-empty and relative: {relative!r}")
    candidate = (root / relative).resolve()
    root_resolved = root.resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ValueError(f"path escapes workspace: {relative}")
    return candidate


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_code(output: str, language: str | None = None) -> str:
    """Extract a code candidate without awarding points for presentation."""
    fence_pattern = re.compile(r"```([\w+#.-]*)\s*\n(.*?)```", re.DOTALL)
    matches = fence_pattern.findall(output)
    if matches:
        preferred = []
        for tag, body in matches:
            tag_norm = tag.lower().strip()
            if not language or tag_norm in {language.lower(), _language_tag(language)}:
                preferred.append(body)
        return (preferred or [body for _, body in matches])[0].strip()
    return output.strip()


def _language_tag(language: str) -> str:
    return {
        "typescript": "ts",
        "javascript": "js",
        "python": "python",
        "cpp": "cpp",
        "c++": "cpp",
    }.get(language.lower(), language.lower())


def extract_json_object(output: str) -> dict[str, Any]:
    text = output.strip()
    if text.startswith("```"):
        text = extract_code(text)
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("no valid JSON object found")


def _copy_workspace(source: Path, destination: Path) -> None:
    for candidate in source.rglob("*"):
        if candidate.is_symlink():
            raise ValueError(f"workspace symlinks are forbidden: {candidate.relative_to(source)}")
    shutil.copytree(source, destination, dirs_exist_ok=True, symlinks=True)


def apply_operations(
    root: Path,
    operations: list[dict[str, Any]],
    *,
    max_operations: int = 50,
) -> PatchApplication:
    """Apply structured operations transactionally using a staging copy."""
    if not operations:
        return PatchApplication(False, [], "no operations")
    if len(operations) > max_operations:
        return PatchApplication(False, [], f"operation budget exceeded: {len(operations)}")
    changed: list[str] = []
    with tempfile.TemporaryDirectory(prefix="nova-patch-") as tmp:
        stage = Path(tmp) / "workspace"
        try:
            _copy_workspace(root, stage)
        except ValueError as exc:
            return PatchApplication(False, [], str(exc))
        try:
            for operation in operations:
                action = str(operation.get("action", "")).lower()
                relative = str(operation.get("path", ""))
                target = safe_resolve(stage, relative)
                expected_hash = operation.get("expected_sha256")
                if expected_hash:
                    if not target.is_file() or file_sha256(target) != str(expected_hash):
                        raise ValueError(f"file hash mismatch: {relative}")
                if action == "create":
                    if target.exists() and not operation.get("overwrite", False):
                        raise ValueError(f"create target already exists: {relative}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(str(operation.get("content", "")), encoding="utf-8")
                elif action == "replace":
                    if not target.is_file():
                        raise ValueError(f"replace target not found: {relative}")
                    original = target.read_text(encoding="utf-8")
                    search = str(operation.get("search", ""))
                    replacement = str(operation.get("replace", ""))
                    expected_count = int(operation.get("expected_count", 1))
                    count = original.count(search)
                    if not search or count != expected_count:
                        raise ValueError(
                            f"search count mismatch for {relative}: "
                            f"expected {expected_count}, got {count}"
                        )
                    target.write_text(
                        original.replace(search, replacement, expected_count), encoding="utf-8"
                    )
                elif action == "write":
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(str(operation.get("content", "")), encoding="utf-8")
                else:
                    raise ValueError(f"unsupported operation action: {action}")
                changed.append(relative)
        except Exception as exc:
            return PatchApplication(False, [], str(exc))

        for relative in sorted(set(changed)):
            source = safe_resolve(stage, relative)
            destination = safe_resolve(root, relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    return PatchApplication(True, sorted(set(changed)))


def apply_unified_diff(root: Path, diff_text: str) -> PatchApplication:
    """Apply a unified diff transactionally with `git apply`."""
    if not diff_text.strip():
        return PatchApplication(False, [], "empty diff")
    raw_paths = re.findall(r"^(?:\+\+\+|---)\s+([^\t\n]+)", diff_text, flags=re.MULTILINE)
    paths: set[str] = set()
    for raw in raw_paths:
        relative = raw.strip()
        if relative == "/dev/null":
            continue
        if relative.startswith(("a/", "b/")):
            relative = relative[2:]
        safe_resolve(root, relative)
        paths.add(relative)
    if not paths:
        return PatchApplication(False, [], "diff contains no repository paths")
    if shutil.which("git") is None:
        return PatchApplication(False, [], "git executable not found")
    with tempfile.TemporaryDirectory(prefix="nova-diff-") as tmp:
        stage = Path(tmp) / "workspace"
        try:
            _copy_workspace(root, stage)
        except ValueError as exc:
            return PatchApplication(False, [], str(exc))
        proc = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", "-"],
            input=diff_text,
            text=True,
            cwd=stage,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
        if proc.returncode != 0:
            return PatchApplication(False, [], proc.stderr.strip())
        for relative in sorted(paths):
            source = safe_resolve(stage, relative)
            destination = safe_resolve(root, relative)
            if source.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            elif destination.exists():
                destination.unlink()
    return PatchApplication(True, sorted(paths))
