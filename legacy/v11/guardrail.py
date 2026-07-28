#!/usr/bin/env python3
"""
guardrail.py — Orchestrator-Level Scope & File-Count Guardrail for Nova 3B

Sits between the Ceiling node and the InternNode in the pipeline.
Implements multiple protection layers:

  PRE-CHECK  — Before Nova runs:
    • If scope_level == "vague", reject immediately and re-route to Ceiling
      for clarification. Nova never sees vague architectural prompts.

  POST-CHECK — After Nova generates output:
    • Count actual # filepath: declarations in <<FILES>> block.
    • Compare against expected_files tagged by the Ceiling.
    • If count doesn't match → auto-reject and re-route.

  FUNCTION NAME CHECK — After Nova generates output:
    • If the prompt or test command references a specific function/symbol name,
      grep the <<FILES>> block for that exact name.
    • Reject and retry if the expected name is missing.

  SCHEMA CHECK — After Nova generates output:
    • Validate output contains <<THINKING>> + (<<FILES>> or <<CLARIFICATION>> or <<RESPONSE>>).
    • Reject raw diffs, freeform text, or any output not matching the trained schema.

This neutralises all identified failure modes without touching model weights:
  - Category 3 (multi-file collapse): detected by file count mismatch.
  - Category 4 (silent hallucination on vague prompts): blocked at PRE-CHECK.
  - Function name hallucination: detected by function name check.
  - Long-context format break: detected by schema check.

Part of the Nova model family by Amaura.
"""

import re
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Set

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

MAX_REROUTES = 2          # Max re-routes per original request before escalating
VAGUE_SCOPE_LEVEL = "vague"
ATOMIC_SCOPE_LEVEL = "atomic"
MULTI_FILE_SCOPE_LEVEL = "multi_file"

VALID_SCOPE_LEVELS = {ATOMIC_SCOPE_LEVEL, MULTI_FILE_SCOPE_LEVEL, VAGUE_SCOPE_LEVEL}

# Regex to count filepath declarations in model output (supports #, //, <!-- -->, /* */, or loose text)
FILEPATH_MARKER = re.compile(r'(?:#|//|<!--|\/\*|^|\s)filepath\s*:', re.IGNORECASE)

# Regex to detect <<CLARIFICATION>> or <<RESPONSE>> block (non-code output formats)
CLARIFICATION_BLOCK = re.compile(r'<<CLARIFICATION>>|<<RESPONSE>>', re.IGNORECASE)

# Regex to extract function/symbol names from prompts and test assertions
FUNC_NAME_IN_PROMPT = re.compile(r'''
    (?:Name\s+(?:the\s+)?function\s+`?(\w+)`?)          |  # "Name the function X"
    (?:named?\s+`(\w+)`)                                  |  # "named X" or "name `X`"
    (?:call\s+it\s+`?(\w+)`?)                             |  # "call it X"
    (?:function\s+should\s+be\s+named\s+`?(\w+)`?)        |  # "should be named X"
    (?:assert\s+(\w+)\s*\()                               |  # "assert func_name("
    (?:def\s+(\w+)\s*\()                                     # "def func_name("
''', re.VERBOSE | re.IGNORECASE)

# Schema markers that define valid Nova output
SCHEMA_THINKING = re.compile(r'<<THINKING>>', re.IGNORECASE)
SCHEMA_FILES = re.compile(r'<<FILES>>', re.IGNORECASE)
SCHEMA_CLARIFICATION = re.compile(r'<<CLARIFICATION>>', re.IGNORECASE)
SCHEMA_RESPONSE = re.compile(r'<<RESPONSE>>', re.IGNORECASE)
SCHEMA_TEST = re.compile(r'<<TEST_COMMAND>>', re.IGNORECASE)


# ═══════════════════════════════════════════════════════════════════════════════
# Verdict Types
# ═══════════════════════════════════════════════════════════════════════════════

class VerdictType(Enum):
    PASS = "PASS"                          # Output is valid, pass through
    REJECT_VAGUE_SCOPE = "REJECT_VAGUE"   # Pre-check: scope is vague/underspecified
    REJECT_FILE_COUNT = "REJECT_COUNT"     # Post-check: file count mismatch
    REJECT_MISSING_SCOPE_TAG = "REJECT_NO_TAG"  # Ceiling failed to tag the task
    REJECT_FUNC_NAME = "REJECT_FUNC_NAME" # Post-check: expected function name missing
    REJECT_SCHEMA = "REJECT_SCHEMA"       # Post-check: output doesn't match trained schema
    REJECT_INJECTION = "REJECT_INJECTION" # Pre-check: prompt injection detected
    REJECT_THINKING_MISMATCH = "REJECT_THINKING_MISMATCH"  # Post-check: <<THINKING>> and <<FILES>> disagree
    ESCALATE = "ESCALATE"                  # Reroute budget exhausted, escalate


@dataclass
class GuardrailVerdict:
    """Result of a guardrail check."""
    type: VerdictType
    passed: bool
    task_id: int
    reason: str = ""
    expected_files: int = 0
    actual_files: int = 0
    scope_level: str = ""
    reroute_count: int = 0
    checked_at: float = field(default_factory=time.time)

    def __str__(self) -> str:
        if self.passed:
            return f"[GUARDRAIL ✅ PASS] task={self.task_id} scope={self.scope_level} files={self.actual_files}/{self.expected_files}"
        return (
            f"[GUARDRAIL ❌ {self.type.value}] task={self.task_id} "
            f"scope={self.scope_level} files={self.actual_files}/{self.expected_files} "
            f"reroutes={self.reroute_count} reason='{self.reason}'"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Guardrail Core
# ═══════════════════════════════════════════════════════════════════════════════

class TaskGuardrail:
    """
    Orchestrator-level guardrail between the Ceiling and Intern nodes.

    Usage in pipeline:
        guardrail = TaskGuardrail()

        # Before Intern runs:
        pre = guardrail.pre_check(task)
        if not pre.passed:
            handle_rejection(pre)
            continue

        # After Intern generates:
        post = guardrail.post_check(task, nova_output_text)
        if not post.passed:
            handle_rejection(post)
            continue

        # If passed both — proceed normally
    """

    def __init__(self, max_reroutes: int = MAX_REROUTES):
        self.max_reroutes = max_reroutes
        # Per task-id reroute counter
        self._reroute_counts: Dict[int, int] = {}

    # ─── Pre-Check ───────────────────────────────────────────────────────────

    def pre_check(self, task) -> GuardrailVerdict:
        """
        Pre-execution check. Runs BEFORE handing task to InternNode.

        Args:
            task: AtomicTask dataclass (must have .expected_files and .scope_level)

        Returns:
            GuardrailVerdict — if .passed is False, do NOT send to Nova.
        """
        task_id = getattr(task, 'id', 0)
        scope_level = getattr(task, 'scope_level', '').lower().strip()
        expected_files = getattr(task, 'expected_files', -1)

        # 1. Check scope tag is present
        if not scope_level or scope_level not in VALID_SCOPE_LEVELS:
            verdict = GuardrailVerdict(
                type=VerdictType.REJECT_MISSING_SCOPE_TAG,
                passed=False,
                task_id=task_id,
                scope_level=scope_level,
                expected_files=expected_files,
                reason=f"Ceiling did not tag scope_level. Got: '{scope_level}'. "
                       f"Valid: {VALID_SCOPE_LEVELS}. Re-routing for retag.",
                reroute_count=self._increment_reroute(task_id),
            )
            logger.warning(str(verdict))
            return verdict

        # 2. Block vague/architectural tasks before Nova even runs
        if scope_level == VAGUE_SCOPE_LEVEL:
            verdict = GuardrailVerdict(
                type=VerdictType.REJECT_VAGUE_SCOPE,
                passed=False,
                task_id=task_id,
                scope_level=scope_level,
                expected_files=0,
                actual_files=0,
                reason="Task is underspecified (scope_level=vague). Nova would hallucinate. "
                       "Re-routing to Ceiling for clarification.",
                reroute_count=self._increment_reroute(task_id),
            )
            logger.warning(str(verdict))
            return verdict

        # 3. Validate expected_files is set and sensible
        if expected_files < 1:
            verdict = GuardrailVerdict(
                type=VerdictType.REJECT_MISSING_SCOPE_TAG,
                passed=False,
                task_id=task_id,
                scope_level=scope_level,
                expected_files=expected_files,
                reason=f"expected_files={expected_files} is invalid (must be >= 1 for non-vague tasks). "
                       f"Re-routing for retag.",
                reroute_count=self._increment_reroute(task_id),
            )
            logger.warning(str(verdict))
            return verdict

        # PRE-CHECK PASSED
        verdict = GuardrailVerdict(
            type=VerdictType.PASS,
            passed=True,
            task_id=task_id,
            scope_level=scope_level,
            expected_files=expected_files,
        )
        logger.info(str(verdict))
        return verdict

    # ─── Post-Check ──────────────────────────────────────────────────────────

    def post_check(self, task, nova_output: str) -> GuardrailVerdict:
        """
        Post-generation check. Runs AFTER InternNode generates output.

        Args:
            task: AtomicTask dataclass
            nova_output: Raw text output from Nova model

        Returns:
            GuardrailVerdict — if .passed is False, reject Nova's output.
        """
        task_id = getattr(task, 'id', 0)
        scope_level = getattr(task, 'scope_level', '').lower().strip()
        expected_files = getattr(task, 'expected_files', 1)

        # If Nova output is a clarification response, that's only valid for
        # tasks that somehow slipped through as vague — treat as pass-through
        # but flag it for logging.
        if CLARIFICATION_BLOCK.search(nova_output):
            verdict = GuardrailVerdict(
                type=VerdictType.PASS,
                passed=True,
                task_id=task_id,
                scope_level=scope_level,
                expected_files=expected_files,
                actual_files=0,
                reason="Nova returned a <<CLARIFICATION>> block (not code). "
                       "Passing through — this is the desired refusal behavior.",
            )
            logger.info(str(verdict))
            return verdict

        # Count actual file declarations
        actual_files = self.count_file_declarations(nova_output)

        # File count mismatch check:
        # A true mismatch occurs when:
        # 1. No files were produced when files were expected (expected_files >= 1 and actual_files == 0)
        # 2. Multi-file collapse: expected_files > 1 and actual_files < expected_files
        # Note: If actual_files >= expected_files and actual_files >= 1, Nova generated all expected files.
        is_count_mismatch = False
        if expected_files >= 1 and actual_files == 0:
            is_count_mismatch = True
        elif expected_files > 1 and actual_files < expected_files:
            is_count_mismatch = True

        if is_count_mismatch:
            reroute_count = self._increment_reroute(task_id)

            # Check if reroute budget is exhausted
            if reroute_count > self.max_reroutes:
                verdict = GuardrailVerdict(
                    type=VerdictType.ESCALATE,
                    passed=False,
                    task_id=task_id,
                    scope_level=scope_level,
                    expected_files=expected_files,
                    actual_files=actual_files,
                    reason=f"File count mismatch after {reroute_count} reroutes "
                           f"(max={self.max_reroutes}). Escalating to frontier/human.",
                    reroute_count=reroute_count,
                )
            else:
                verdict = GuardrailVerdict(
                    type=VerdictType.REJECT_FILE_COUNT,
                    passed=False,
                    task_id=task_id,
                    scope_level=scope_level,
                    expected_files=expected_files,
                    actual_files=actual_files,
                    reason=f"File count mismatch: expected {expected_files}, "
                           f"got {actual_files}. Nova likely collapsed multi-file "
                           f"output into a single block. Re-routing to Ceiling.",
                    reroute_count=reroute_count,
                )
            logger.warning(str(verdict))
            return verdict

        # POST-CHECK PASSED
        verdict = GuardrailVerdict(
            type=VerdictType.PASS,
            passed=True,
            task_id=task_id,
            scope_level=scope_level,
            expected_files=expected_files,
            actual_files=actual_files,
        )
        logger.info(str(verdict))
        return verdict

    # ─── Utilities ───────────────────────────────────────────────────────────

    @staticmethod
    def count_file_declarations(text: str) -> int:
        """
        Count the number of file declarations in Nova output.
        Uses NovaOutputParser to parse actual file actions, falling back to regex markers.

        Args:
            text: Raw Nova model output

        Returns:
            Number of file declarations found.
        """
        try:
            from output_parser import NovaOutputParser
            parsed = NovaOutputParser().parse(text)
            if parsed.files:
                return len(parsed.files)
        except Exception:
            pass
        return len(FILEPATH_MARKER.findall(text))

    def reroute_count_for(self, task_id: int) -> int:
        """Return the current reroute count for a given task."""
        return self._reroute_counts.get(task_id, 0)

    def reset_reroute_count(self, task_id: int):
        """Reset reroute count (e.g., after a successful pass)."""
        self._reroute_counts.pop(task_id, None)

    def _increment_reroute(self, task_id: int) -> int:
        count = self._reroute_counts.get(task_id, 0) + 1
        self._reroute_counts[task_id] = count
        return count

    # ─── Function Name Check (P1) ────────────────────────────────────────────

    @staticmethod
    def extract_expected_func_names(prompt: str) -> Set[str]:
        """
        Extract function/symbol names that the prompt explicitly expects
        the model to use. Returns a set of expected names.

        Matches patterns like:
          - "Name the function `find_Volume`"
          - "assert find_Volume(10,8,6) == 240"
          - "def find_Volume(base, height, length):"
          - "call it `remove_dirty_chars`"
          - "function should be named `test_duplicate`"
        """
        names = set()
        for match in FUNC_NAME_IN_PROMPT.finditer(prompt):
            # Each group in the alternation captures one pattern
            for group in match.groups():
                if group:
                    names.add(group)
        # Filter out common false positives
        false_positives = {'test_solution', 'src_solution', 'assert_equal',
                          'test_auth', 'test_cache', 'test_api', 'test_utils'}
        names -= false_positives
        return names

    def function_name_check(
        self, task, nova_output: str, prompt: str = ""
    ) -> GuardrailVerdict:
        """
        Post-generation check: verify that expected function names from the
        prompt actually appear in the generated <<FILES>> block.

        This catches the silent failure where the model writes correct code
        but with the WRONG function name.

        Args:
            task: AtomicTask dataclass
            nova_output: Raw text output from Nova model
            prompt: The original prompt sent to Nova (for name extraction)

        Returns:
            GuardrailVerdict — if .passed is False, reject Nova's output.
        """
        task_id = getattr(task, 'id', 0)
        scope_level = getattr(task, 'scope_level', '').lower().strip()
        expected_files = getattr(task, 'expected_files', 1)

        # Skip check if output is a clarification/response (no code to check)
        if CLARIFICATION_BLOCK.search(nova_output):
            return GuardrailVerdict(
                type=VerdictType.PASS, passed=True, task_id=task_id,
                scope_level=scope_level, expected_files=expected_files,
                reason="Non-code response — function name check skipped.",
            )

        # Use prompt from task description if not provided separately
        if not prompt:
            prompt = getattr(task, 'description', '')

        expected_names = self.extract_expected_func_names(prompt)

        if not expected_names:
            # No specific function name expected — skip check
            return GuardrailVerdict(
                type=VerdictType.PASS, passed=True, task_id=task_id,
                scope_level=scope_level, expected_files=expected_files,
                reason="No explicit function name in prompt — check skipped.",
            )

        # Check if ALL expected function names appear in the output
        missing_names = []
        for name in expected_names:
            # Look for 'def name(' or the name anywhere in the output
            if not re.search(rf'\bdef\s+{re.escape(name)}\s*\(', nova_output):
                if name not in nova_output:
                    missing_names.append(name)

        if missing_names:
            reroute_count = self._increment_reroute(task_id)
            if reroute_count > self.max_reroutes:
                return GuardrailVerdict(
                    type=VerdictType.ESCALATE, passed=False, task_id=task_id,
                    scope_level=scope_level, expected_files=expected_files,
                    reason=f"Function name mismatch after {reroute_count} reroutes. "
                           f"Missing: {missing_names}. Escalating.",
                    reroute_count=reroute_count,
                )
            return GuardrailVerdict(
                type=VerdictType.REJECT_FUNC_NAME, passed=False, task_id=task_id,
                scope_level=scope_level, expected_files=expected_files,
                reason=f"Expected function name(s) {missing_names} not found in output. "
                       f"Model likely hallucinated a different name. Rejecting for retry.",
                reroute_count=reroute_count,
            )

        return GuardrailVerdict(
            type=VerdictType.PASS, passed=True, task_id=task_id,
            scope_level=scope_level, expected_files=expected_files,
            reason=f"All expected function names found: {expected_names}",
        )

    # ─── Schema Check (P3) ───────────────────────────────────────────────────

    def schema_check(self, task, nova_output: str) -> GuardrailVerdict:
        """
        Post-generation check: verify that the output matches the trained
        Nova schema (<<THINKING>> + <<FILES>>/<<CLARIFICATION>>/<<RESPONSE>>).

        Catches long-context format breaks where the model emits raw diffs
        or freeform text instead of the trained schema.

        Args:
            task: AtomicTask dataclass
            nova_output: Raw text output from Nova model

        Returns:
            GuardrailVerdict — if .passed is False, reject Nova's output.
        """
        task_id = getattr(task, 'id', 0)
        scope_level = getattr(task, 'scope_level', '').lower().strip()
        expected_files = getattr(task, 'expected_files', 1)

        has_thinking = bool(SCHEMA_THINKING.search(nova_output))
        has_files = bool(SCHEMA_FILES.search(nova_output))
        has_clarification = bool(SCHEMA_CLARIFICATION.search(nova_output))
        has_response = bool(SCHEMA_RESPONSE.search(nova_output))
        has_body = has_files or has_clarification or has_response

        if has_thinking and has_body:
            return GuardrailVerdict(
                type=VerdictType.PASS, passed=True, task_id=task_id,
                scope_level=scope_level, expected_files=expected_files,
                reason="Output matches trained schema.",
            )

        # Schema violation
        missing = []
        if not has_thinking:
            missing.append("<<THINKING>>")
        if not has_body:
            missing.append("<<FILES>> or <<CLARIFICATION>> or <<RESPONSE>>")

        reroute_count = self._increment_reroute(task_id)
        if reroute_count > self.max_reroutes:
            return GuardrailVerdict(
                type=VerdictType.ESCALATE, passed=False, task_id=task_id,
                scope_level=scope_level, expected_files=expected_files,
                reason=f"Schema violation after {reroute_count} reroutes. "
                       f"Missing: {missing}. Escalating.",
                reroute_count=reroute_count,
            )

        return GuardrailVerdict(
            type=VerdictType.REJECT_SCHEMA, passed=False, task_id=task_id,
            scope_level=scope_level, expected_files=expected_files,
            reason=f"Output does not match trained schema. Missing: {missing}. "
                   f"Got raw text/diff instead of structured output. Rejecting for retry.",
            reroute_count=reroute_count,
        )


    # ─── Thinking/Files Consistency Check (P3) ────────────────────────────

    def thinking_files_consistency_check(
        self, task, nova_output: str
    ) -> GuardrailVerdict:
        """
        Post-generation check: verify that <<THINKING>> and <<FILES>> are
        consistent. Catches the failure mode where THINKING correctly plans
        file creation but <<FILES>> emits 'none', empty, or whitespace.

        Args:
            task: AtomicTask dataclass
            nova_output: Raw text output from Nova model

        Returns:
            GuardrailVerdict — if .passed is False, reject Nova's output.
        """
        task_id = getattr(task, 'id', 0)
        scope_level = getattr(task, 'scope_level', '').lower().strip()
        expected_files = getattr(task, 'expected_files', 1)

        # Skip check if output is a clarification/response (no code expected)
        if CLARIFICATION_BLOCK.search(nova_output):
            return GuardrailVerdict(
                type=VerdictType.PASS, passed=True, task_id=task_id,
                scope_level=scope_level, expected_files=expected_files,
                reason="Non-code response — thinking/files check skipped.",
            )

        # 1. Extract <<THINKING>> block
        thinking_match = SCHEMA_THINKING.search(nova_output)
        if not thinking_match:
            # No thinking block — let schema_check handle this
            return GuardrailVerdict(
                type=VerdictType.PASS, passed=True, task_id=task_id,
                scope_level=scope_level, expected_files=expected_files,
                reason="No <<THINKING>> block found — deferring to schema_check.",
            )

        # Get the thinking text (everything between <<THINKING>> and the next block)
        thinking_start = thinking_match.end()
        # Find the next block delimiter after <<THINKING>>
        next_block = re.search(
            r'<<(?:FILES|CLARIFICATION|RESPONSE|TEST_COMMAND)>>',
            nova_output[thinking_start:], re.IGNORECASE
        )
        if next_block:
            thinking_text = nova_output[thinking_start:thinking_start + next_block.start()].strip()
        else:
            thinking_text = nova_output[thinking_start:].strip()

        # 2. Detect file-creation intent in <<THINKING>>
        FILE_INTENT_PATTERNS = [
            re.compile(r'[Cc]reating\s+\d+\s+file', re.IGNORECASE),
            re.compile(r'[Cc]reating\s+file', re.IGNORECASE),
            re.compile(r'I\s+will\s+create', re.IGNORECASE),
            re.compile(r'I\'ll\s+(?:create|write|implement)', re.IGNORECASE),
            re.compile(r'[Cc]reating\s+[`\']?\w+[/\\]\w+', re.IGNORECASE),  # Creating src/file.py
            re.compile(r'[Ww]riting\s+(?:the\s+)?(?:code|solution|implementation)', re.IGNORECASE),
            re.compile(r'`[\w/]+\.\w{1,5}`', re.IGNORECASE),  # backtick-quoted filepath
        ]

        thinking_mentions_files = any(
            pattern.search(thinking_text) for pattern in FILE_INTENT_PATTERNS
        )

        if not thinking_mentions_files:
            # Thinking doesn't mention creating files — no mismatch possible
            return GuardrailVerdict(
                type=VerdictType.PASS, passed=True, task_id=task_id,
                scope_level=scope_level, expected_files=expected_files,
                reason="<<THINKING>> does not mention file creation — no mismatch.",
            )

        # 3. Check <<FILES>> block for degenerate content
        files_match = SCHEMA_FILES.search(nova_output)
        if not files_match:
            # <<FILES>> block entirely missing despite thinking mentioning files
            reroute_count = self._increment_reroute(task_id)
            if reroute_count > self.max_reroutes:
                return GuardrailVerdict(
                    type=VerdictType.ESCALATE, passed=False, task_id=task_id,
                    scope_level=scope_level, expected_files=expected_files,
                    reason=f"Thinking/Files mismatch after {reroute_count} reroutes. Escalating.",
                    reroute_count=reroute_count,
                )
            return GuardrailVerdict(
                type=VerdictType.REJECT_THINKING_MISMATCH, passed=False, task_id=task_id,
                scope_level=scope_level, expected_files=expected_files, actual_files=0,
                reason="<<THINKING>> plans file creation but <<FILES>> block is missing entirely.",
                reroute_count=reroute_count,
            )

        # Get content between <<FILES>> and the next block (<<TEST_COMMAND>> or end)
        files_start = files_match.end()
        next_block_after_files = re.search(
            r'<<TEST_COMMAND>>',
            nova_output[files_start:], re.IGNORECASE
        )
        if next_block_after_files:
            files_content = nova_output[files_start:files_start + next_block_after_files.start()].strip()
        else:
            files_content = nova_output[files_start:].strip()

        # Check if <<FILES>> content is degenerate (empty, 'none', 'null', whitespace-only)
        DEGENERATE_VALUES = {'none', 'null', 'n/a', 'na', 'empty', ''}
        is_degenerate = files_content.lower().strip() in DEGENERATE_VALUES

        # Also check: has <<FILES>> but zero actual filepath declarations
        actual_file_count = self.count_file_declarations(nova_output)

        if is_degenerate or (thinking_mentions_files and actual_file_count == 0):
            files_state = (
                f'degenerate ("{files_content[:30]}")'
                if is_degenerate
                else "empty (0 filepath declarations)"
            )
            reroute_count = self._increment_reroute(task_id)
            if reroute_count > self.max_reroutes:
                return GuardrailVerdict(
                    type=VerdictType.ESCALATE, passed=False, task_id=task_id,
                    scope_level=scope_level, expected_files=expected_files,
                    reason=f"Thinking/Files mismatch after {reroute_count} reroutes. Escalating.",
                    reroute_count=reroute_count,
                )
            return GuardrailVerdict(
                type=VerdictType.REJECT_THINKING_MISMATCH, passed=False, task_id=task_id,
                scope_level=scope_level, expected_files=expected_files,
                actual_files=actual_file_count,
                reason=f"<<THINKING>> plans file creation but <<FILES>> is "
                       f"{files_state}. Internal contradiction — model reasoned correctly "
                       "but failed to emit output.",
                reroute_count=reroute_count,
            )

        # CONSISTENCY CHECK PASSED
        return GuardrailVerdict(
            type=VerdictType.PASS, passed=True, task_id=task_id,
            scope_level=scope_level, expected_files=expected_files,
            actual_files=actual_file_count,
            reason=f"<<THINKING>> and <<FILES>> are consistent ({actual_file_count} files planned and emitted).",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Input Sanitizer — Prompt Injection Defense (Priority 1)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SanitizeResult:
    """Result of input sanitization."""
    original_prompt: str
    sanitized_prompt: str
    is_rejected: bool = False           # Hard-reject: don't send to model at all
    is_modified: bool = False           # Soft: prompt was neutralized but still usable
    injection_patterns_found: List[str] = field(default_factory=list)
    dangerous_payloads_found: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        if self.is_rejected:
            return (f"[SANITIZER ❌ REJECTED] Dangerous payloads: {self.dangerous_payloads_found} "
                    f"Injection patterns: {self.injection_patterns_found}")
        if self.is_modified:
            return (f"[SANITIZER ⚠️ NEUTRALIZED] Injection patterns: {self.injection_patterns_found} "
                    f"Warnings: {self.warnings}")
        return "[SANITIZER ✅ CLEAN]"


class InputSanitizer:
    """
    Orchestrator-level input sanitization for prompt injection defense.

    Sits BEFORE the Intern node in the pipeline. Detects and neutralizes
    prompt injection attempts embedded in task descriptions.

    Strategy:
      - HARD REJECT: Prompts containing dangerous executable payloads
        (rm -rf, system commands embedded in <<TEST_COMMAND>> format)
      - NEUTRALIZE: Softer injection attempts ("ignore previous instructions")
        are wrapped in [UNTRUSTED_INPUT: "..."] markers so Nova sees them as
        data, not instructions
      - PASS: Clean prompts are returned unchanged

    This does NOT rely on the model to self-defend — it operates at the
    system layer before the model ever sees the input.
    """

    # ── Soft injection patterns: neutralize but don't reject ──────────────
    INJECTION_PATTERNS = [
        re.compile(r'ignore\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+instructions?', re.IGNORECASE),
        re.compile(r'ignore\s+everything\s+(?:above|before|and)', re.IGNORECASE),
        re.compile(r'forget\s+(?:everything|this|all|previous|the\s+above)', re.IGNORECASE),
        re.compile(r'disregard\s+(?:above|previous|all|prior|the)', re.IGNORECASE),
        re.compile(r'you\s+are\s+now\b', re.IGNORECASE),
        re.compile(r'new\s+instructions?\s*:', re.IGNORECASE),
        re.compile(r'system\s+(?:override|prompt|message)\s*:', re.IGNORECASE),
        re.compile(r'act\s+as\s+(?:if|though)\s+you', re.IGNORECASE),
        re.compile(r'pretend\s+(?:you\s+are|to\s+be|that)', re.IGNORECASE),
        re.compile(r'instead\s+(?:of|,)\s*(?:do|write|output|print)', re.IGNORECASE),
        re.compile(r"print\s+['\"]hacked['\"]", re.IGNORECASE),
        re.compile(r'(?:output|emit|write|print)\s+[\'\"]?<<(?:TEST_COMMAND|FILES|THINKING|RESPONSE)>>', re.IGNORECASE),
    ]

    # ── Hard rejection patterns: dangerous payloads that must never reach the model ──
    DANGEROUS_PAYLOAD_PATTERNS = [
        re.compile(r'<<TEST_COMMAND>>\s*(?:rm|del|format|mkfs|dd|:fork|shutdown|reboot)', re.IGNORECASE),
        re.compile(r'rm\s+-rf\s+/', re.IGNORECASE),
        re.compile(r'rm\s+-rf\s+~', re.IGNORECASE),
        re.compile(r'mkfs\.', re.IGNORECASE),
        re.compile(r'dd\s+if=/dev/(?:zero|random)\s+of=/', re.IGNORECASE),
        re.compile(r':(){ :\|:& };:', re.IGNORECASE),  # fork bomb
        re.compile(r'curl\s+.*\|\s*(?:bash|sh|zsh)', re.IGNORECASE),  # pipe-to-shell
        re.compile(r'wget\s+.*\|\s*(?:bash|sh|zsh)', re.IGNORECASE),
        re.compile(r'eval\s*\(.*\bfetch\b', re.IGNORECASE),  # eval(fetch(...))
    ]

    # ── Patterns that look like injection but are legitimate in coding contexts ──
    FALSE_POSITIVE_GUARDS = [
        # "ignore whitespace" / "ignore case" / "ignore errors" are normal coding phrases
        re.compile(r'ignore\s+(?:whitespace|case|errors?|warnings?|null|empty|blank|duplicates?|order)', re.IGNORECASE),
        # "forget" in caching context
        re.compile(r'forget\s+(?:cache|session|token|key)', re.IGNORECASE),
    ]

    def __init__(self):
        pass

    def sanitize(self, prompt: str) -> SanitizeResult:
        """
        Sanitize an input prompt before it reaches the Nova model.

        Args:
            prompt: Raw input prompt (from Ceiling decomposition or user)

        Returns:
            SanitizeResult with the sanitized prompt and metadata.
        """
        result = SanitizeResult(
            original_prompt=prompt,
            sanitized_prompt=prompt,
        )

        if not prompt or not prompt.strip():
            return result

        # ── Step 1: Check for dangerous payloads (hard reject) ────────────
        for pattern in self.DANGEROUS_PAYLOAD_PATTERNS:
            match = pattern.search(prompt)
            if match:
                result.dangerous_payloads_found.append(match.group(0))

        if result.dangerous_payloads_found:
            result.is_rejected = True
            result.warnings.append(
                f"HARD REJECT: Dangerous payload(s) detected: {result.dangerous_payloads_found}. "
                f"Prompt will NOT be sent to the model."
            )
            logger.warning(str(result))
            return result

        # ── Step 2: Check for soft injection patterns (neutralize) ────────
        working_prompt = prompt
        for pattern in self.INJECTION_PATTERNS:
            match = pattern.search(working_prompt)
            if match:
                matched_text = match.group(0)

                # Check false positive guards before flagging
                is_false_positive = any(
                    fp.search(working_prompt[max(0, match.start()-10):match.end()+30])
                    for fp in self.FALSE_POSITIVE_GUARDS
                )

                if is_false_positive:
                    continue

                result.injection_patterns_found.append(matched_text)
                # Neutralize: wrap the injected segment in inert markers
                working_prompt = working_prompt.replace(
                    matched_text,
                    f'[UNTRUSTED_INPUT: "{matched_text}"]'
                )

        if result.injection_patterns_found:
            result.is_modified = True
            result.sanitized_prompt = working_prompt
            result.warnings.append(
                f"Injection pattern(s) neutralized: {result.injection_patterns_found}. "
                f"Injected segments wrapped in [UNTRUSTED_INPUT] markers."
            )
            logger.warning(str(result))
        else:
            result.sanitized_prompt = prompt

        return result


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience: build a reroute-to-ceiling message
# ═══════════════════════════════════════════════════════════════════════════════

def build_reroute_message(verdict: GuardrailVerdict, original_task_description: str) -> str:
    """
    Build a structured re-prompt to send back to the Ceiling node
    when the guardrail rejects Nova's output.

    Args:
        verdict: The failing GuardrailVerdict
        original_task_description: The task description that was rejected

    Returns:
        String to send back to CeilingNode.decompose() as input.
    """
    if verdict.type == VerdictType.REJECT_VAGUE_SCOPE:
        return (
            f"[GUARDRAIL REROUTE — VAGUE SCOPE]\n"
            f"The following task was rejected because it is underspecified:\n\n"
            f"  '{original_task_description}'\n\n"
            f"Please ask the user for the following before decomposing:\n"
            f"- What specific files should be created or modified?\n"
            f"- What language/stack is required?\n"
            f"- What is the concrete scope (single function, module, full service)?\n\n"
            f"Do NOT attempt to decompose until clarification is provided."
        )

    if verdict.type == VerdictType.REJECT_FILE_COUNT:
        return (
            f"[GUARDRAIL REROUTE — FILE COUNT MISMATCH]\n"
            f"Nova generated {verdict.actual_files} file(s) but {verdict.expected_files} "
            f"were expected for task:\n\n"
            f"  '{original_task_description}'\n\n"
            f"Please re-decompose this task into {verdict.expected_files} atomic sub-tasks, "
            f"one per file. Each sub-task must target exactly 1 file.\n"
            f"Re-tag each sub-task with expected_files=1 and scope_level=atomic."
        )

    if verdict.type == VerdictType.REJECT_MISSING_SCOPE_TAG:
        return (
            f"[GUARDRAIL REROUTE — MISSING SCOPE TAG]\n"
            f"Task was missing a valid scope_level or expected_files tag:\n\n"
            f"  '{original_task_description}'\n\n"
            f"Please re-emit the task JSON with:\n"
            f"  scope_level: 'atomic' | 'multi_file' | 'vague'\n"
            f"  expected_files: <integer >= 1, or 0 for vague>"
        )

    if verdict.type == VerdictType.REJECT_FUNC_NAME:
        return (
            f"[GUARDRAIL REROUTE — FUNCTION NAME MISMATCH]\n"
            f"Nova generated code but used the WRONG function name for task:\n\n"
            f"  '{original_task_description}'\n\n"
            f"Reason: {verdict.reason}\n"
            f"Please ensure your prompt explicitly includes the expected function name "
            f"with phrasing like 'Name the function `X`' and retry."
        )

    if verdict.type == VerdictType.REJECT_SCHEMA:
        return (
            f"[GUARDRAIL REROUTE — SCHEMA VIOLATION]\n"
            f"Nova's output did not match the trained <<THINKING>>/<<FILES>>/<<TEST_COMMAND>> schema:\n\n"
            f"  '{original_task_description}'\n\n"
            f"Reason: {verdict.reason}\n"
            f"Re-prompt with explicit format instructions and retry."
        )

    if verdict.type == VerdictType.REJECT_INJECTION:
        return (
            f"[GUARDRAIL REJECT — PROMPT INJECTION DETECTED]\n"
            f"The following prompt was blocked because it contains injection patterns:\n\n"
            f"  '{original_task_description}'\n\n"
            f"Reason: {verdict.reason}\n"
            f"This prompt was NOT sent to Nova. Review and sanitize the input source."
        )

    if verdict.type == VerdictType.REJECT_THINKING_MISMATCH:
        return (
            f"[GUARDRAIL REROUTE — THINKING/FILES MISMATCH]\n"
            f"Nova's <<THINKING>> block planned file creation but <<FILES>> was empty or 'none':\n\n"
            f"  '{original_task_description}'\n\n"
            f"Reason: {verdict.reason}\n"
            f"This is an internal model contradiction. Retry with the same prompt."
        )

    if verdict.type == VerdictType.ESCALATE:
        return (
            f"[GUARDRAIL ESCALATE — REROUTE BUDGET EXHAUSTED]\n"
            f"Task failed guardrail {verdict.reroute_count} times:\n\n"
            f"  '{original_task_description}'\n\n"
            f"This task requires human review or frontier model intervention. "
            f"Do not retry with Nova."
        )

    return f"[GUARDRAIL REROUTE] Task '{original_task_description}' rejected: {verdict.reason}"


# ═══════════════════════════════════════════════════════════════════════════════
# Standalone test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("\n" + "═" * 60)
    print("  TaskGuardrail — Standalone Test")
    print("═" * 60)

    # Minimal mock AtomicTask for testing
    class MockTask:
        def __init__(self, id, desc, expected_files, scope_level):
            self.id = id
            self.description = desc
            self.expected_files = expected_files
            self.scope_level = scope_level

    guardrail = TaskGuardrail(max_reroutes=2)

    # Test 1: Vague scope → should reject pre-check
    print("\n[Test 1] Vague scope pre-check:")
    t1 = MockTask(1, "Build a scalable cloud-native microservice architecture.", 0, "vague")
    v1 = guardrail.pre_check(t1)
    print(f"  {v1}")
    assert not v1.passed, "Should have rejected vague scope"

    # Test 2: Valid atomic task → pre-check should pass
    print("\n[Test 2] Atomic task pre-check:")
    t2 = MockTask(2, "Create src/cache.py with get/set/delete methods.", 1, "atomic")
    v2 = guardrail.pre_check(t2)
    print(f"  {v2}")
    assert v2.passed, "Should have passed atomic task"

    # Test 3: File count mismatch → post-check should reject
    print("\n[Test 3] File count mismatch post-check:")
    t3 = MockTask(3, "Refactor auth module into 4 files.", 4, "multi_file")
    nova_output_one_file = """<<THINKING>>
I'll put everything into one file.

<<FILES>>
```python
# filepath: src/auth.py
# action: MODIFY

class Auth:
    pass
```

<<TEST_COMMAND>>
pytest test_auth.py
"""
    v3 = guardrail.post_check(t3, nova_output_one_file)
    print(f"  {v3}")
    assert not v3.passed, "Should have rejected — only 1 file instead of 4"

    # Test 4: File count matches → post-check should pass
    print("\n[Test 4] Correct file count post-check:")
    t4 = MockTask(4, "Create src/models.py", 1, "atomic")
    nova_output_correct = """<<THINKING>>
Creating models.py.

<<FILES>>
```python
# filepath: src/models.py
# action: CREATE

class User:
    pass
```

<<TEST_COMMAND>>
pytest tests/
"""
    v4 = guardrail.post_check(t4, nova_output_correct)
    print(f"  {v4}")
    assert v4.passed, "Should have passed — 1 file as expected"

    # Test 5: Missing scope tag
    print("\n[Test 5] Missing scope tag:")
    t5 = MockTask(5, "Do something", -1, "")
    v5 = guardrail.pre_check(t5)
    print(f"  {v5}")
    assert not v5.passed, "Should have rejected — no scope tag"

    # Test 6: Reroute budget exhaustion
    print("\n[Test 6] Reroute budget exhaustion:")
    guardrail2 = TaskGuardrail(max_reroutes=1)
    t6 = MockTask(6, "Refactor into 3 files.", 3, "multi_file")
    bad_output = """<<THINKING>>\nOne file.\n\n<<FILES>>\n```python\n# filepath: merged.py\n```\n\n<<TEST_COMMAND>>\ntest"""
    v6a = guardrail2.post_check(t6, bad_output)  # Reroute 1
    v6b = guardrail2.post_check(t6, bad_output)  # Should escalate
    print(f"  Reroute 1: {v6a.type.value}")
    print(f"  Reroute 2: {v6b.type.value}")
    assert v6b.type == VerdictType.ESCALATE, "Should have escalated after budget exhausted"

    print("\n✅ All guardrail tests passed.")
