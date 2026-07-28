# Nova V12

Nova V12 is Amaura Labs' small, local **atomic coding worker**. Nexus or another
planner supplies a tightly scoped task, at most three allowed files, repository
context, and a test command. Nova returns a strict file-operation object. The
runtime stages the patch in a copy, runs the planner-owned tests, permits one
repair attempt, and commits only a passing patch.

This repository contains the complete V12 data, training, inference, evaluation,
and GGUF export code. It does **not** contain trained V12 weights yet. A checkpoint
becomes V12 only after the foundation bake-off, verified-data build, GPU training,
held-out comparison, and quantized release gates have produced evidence.

## Product boundary

Nova V12 is designed for:

- one atomic bug fix or small feature;
- one to three explicitly allowed files;
- deterministic local inference through Ollama;
- full-file create, update, or delete operations;
- test-gated commit with a single repair attempt;
- escalation when context or scope is insufficient.

It is not a frontier model, architect, autonomous app builder, or replacement for
Nexus planning and validation.

## Protocol

Success:

```json
{
  "schema_version": "nova.patch.v1",
  "summary": "Fix the empty-input boundary",
  "files": [
    {
      "path": "src/parser.py",
      "action": "update",
      "content": "full final file content"
    }
  ]
}
```

Escalation:

```json
{
  "schema_version": "nova.escalation.v1",
  "reason_code": "missing_context",
  "message": "The caller contract is not supplied."
}
```

No Markdown fences, prose, extra fields, absolute paths, path traversal, VCS files,
or more than three changed files are accepted.

## Repository layout

| Path | Purpose |
|---|---|
| `src/nova_v12/` | The single installable V12 runtime and engineering package |
| `src/nova_v12/data/` | Licence, security, quality, deduplication, FIM and split tooling |
| `src/nova_v12/eval/` | Broad foundation bake-off and executable scoring |
| `src/nova_v12/training/` | CPT, verified QLoRA SFT, distillation, DPO and GGUF export |
| `configs/nova-v12.yaml` | Pilot and release thresholds |
| `tests/` | Fail-closed unit and integration tests |
| `legacy/v11/` | Preserved V11 code and evidence; excluded from V12 packaging |
| `legacy/v11/coding_agent/nexus/nova_v12_backend.py` | V12 adapter for the archived Nexus checkout |

The permissive bake-off protocol exists only to compare untuned foundations across
multiple task styles. The shipped runtime always uses the strict Nova patch or
escalation protocol. Pre-V12 experiment files remain available for audit history,
but they are not packaged and cannot pass V12 training gates.

## Local development

```bash
python -m pip install -e ".[dev]"
ruff check src/nova_v12 tests legacy/v11/coding_agent/nexus/nova_v12_backend.py
pytest --cov=nova_v12 --cov-report=term-missing
```

## Build verified SFT data

Input records contain the atomic task, repository files before the patch, the
protocol response, licence, split, and source provenance. Every accepted response
must apply and pass its real test command. Repair examples must fail before the
patch.

```bash
nova-v12 build-dataset \
  --source artifacts/source/atomic-patches.jsonl \
  --output-dir artifacts/data/sft

nova-v12 check-dataset \
  --dataset-dir artifacts/data/sft \
  --minimum-verified 100
```

Rejected records block publication by default. Training refuses absent, modified,
unverified, contaminated, or under-sized manifests.

## Build execution-ranked preferences

```bash
nova-v12-preferences \
  --source artifacts/source/preference-candidates.jsonl \
  --output-dir artifacts/data/preferences
```

The chosen patch must pass; the rejected patch must fail to apply or fail tests.
Plausibility labels without execution evidence are not accepted.

## Lock the foundation

Run every base candidate on the same held-out atomic tasks, then lock the measured
winner:

```bash
nova-v12-eval run \
  --cases eval/private/atomic-held-out.jsonl \
  --model candidate-a \
  --revision IMMUTABLE_REVISION \
  --output-dir artifacts/eval/candidate-a

nova-v12-foundation \
  --evaluation artifacts/eval/candidate-a/evaluation.json \
  --evaluation artifacts/eval/candidate-b/evaluation.json \
  --metadata configs/foundation-candidates.json \
  --output artifacts/foundation.lock.json
```

The selector uses task success, protocol validity, patch application, scope safety,
and first-attempt success. It does not award style points for code-looking text.

## Train

Create a content-addressed run plan:

```bash
nova-v12-train \
  --config configs/nova-v12.yaml \
  --foundation-lock artifacts/foundation.lock.json \
  --dataset-dir artifacts/data/sft \
  --run-dir artifacts/runs/v12-pilot-001
```

Execute the SFT and DPO commands recorded in `run.json` on an NVIDIA GPU. The
production recipe uses 4-bit NF4 QLoRA with double quantization and trains only on
completion tokens. Each run records model revision, dataset hash, seed, versions,
and metrics.

## Evaluate and release

Evaluate the untouched foundation, trained BF16/FP16 candidate, and every GGUF
quantization on the identical held-out set:

```bash
nova-v12-eval release \
  --candidate artifacts/eval/nova-v12-q4/evaluation.json \
  --baseline artifacts/eval/foundation/evaluation.json \
  --output artifacts/eval/release.json
```

Release-candidate gates:

- at least 1,000 repository-disjoint, private held-out atomic tasks;
- at least 99% valid protocol;
- at least 97% patch application;
- at least 70% first-attempt success and 80% success after one repair;
- 100% scope compliance and raw-output evidence;
- no silent success when tests fail;
- no more than two percentage points of Q4 degradation;
- M3 peak memory at or below 7.2 GB and a target of at least 25 tokens/second.

Until those gates pass, model cards and launch materials must say
**training candidate**, not released Nova V12.

## Run through Ollama

```bash
nova-v12 run \
  --model nova-v12:q4_k_m \
  --task task.json \
  --workspace /path/to/repository \
  --evidence artifacts/run-evidence.json
```

The model cannot choose the test command. Nexus owns task decomposition, allowed
files, context, tests, approvals, and any broader retry strategy.

## Licence

Repository code is MIT. Model and dataset licensing also depends on the foundation
and every source record; the release process records those separately.
