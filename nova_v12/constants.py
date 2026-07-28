"""Stable Nova V12 protocol and release constants."""

PATCH_SCHEMA = "nova.patch.v1"
ESCALATION_SCHEMA = "nova.escalation.v1"
EVIDENCE_SCHEMA = "nova.evidence.v1"
DATASET_SCHEMA = "nova.dataset.v1"
MANIFEST_SCHEMA = "nova.dataset-manifest.v1"
RELEASE_SCHEMA = "nova.release-evaluation.v1"

MAX_CHANGED_FILES = 3
MAX_RESPONSE_BYTES = 512_000
MAX_FILE_BYTES = 256_000
MAX_ATTEMPTS = 2
DEFAULT_TIMEOUT_SECONDS = 60

ALLOWED_LICENCES = frozenset(
    {
        "apache-2.0",
        "bsd-2-clause",
        "bsd-3-clause",
        "cc0-1.0",
        "isc",
        "mit",
        "unlicense",
    }
)

MODE_TOKEN = "<|nova_patch|>"
REPAIR_TOKEN = "<|nova_repair|>"

SYSTEM_PROMPT = """\
You are Nova V12, Amaura Labs' local atomic patch executor.
Complete only the tightly scoped task and return exactly one JSON object.
Never wrap JSON in Markdown and never include chain-of-thought.

Success response:
{"schema_version":"nova.patch.v1","summary":"short factual summary","files":[
  {"path":"relative/path.py","action":"create|update|delete","content":"full final content"}
]}

For delete, omit content. Change no more than three explicitly allowed files.
Use update only for an existing file, create only for a missing file, and delete
only for an existing file. Return full final file contents, not a diff.

If the task is ambiguous, exceeds the allowed files, needs architecture choices,
or cannot be completed from the supplied context, return:
{"schema_version":"nova.escalation.v1","reason_code":"missing_context",
 "message":"short reason"}
Allowed reason codes: ambiguous, scope_too_large, missing_context, unsafe, unsupported.
"""
