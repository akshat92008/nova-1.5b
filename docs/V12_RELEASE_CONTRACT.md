# Nova V12 Release Contract

## Locked product

Nova V12 is a local atomic patch executor. One task may change no more than three
planner-approved files. The runtime permits one repair attempt and otherwise
escalates without changing the working tree.

## What “built” means

The repository implementation is complete when:

1. protocol and scope enforcement fail closed;
2. dataset and preference records require execution evidence;
3. foundation selection uses identical held-out tasks;
4. training inputs are content-addressed and revision-pinned;
5. candidate patches are verified transactionally;
6. raw outputs and test evidence are retained;
7. release compares candidate and quantizations with the untouched base.

The model is trained only after those controls exist.

## What “released” means

A checkpoint may be named Nova V12 only when `nova-v12-eval release` returns
`release_status: passed` on at least 100 private held-out atomic tasks and the
quantized artifact used by customers was itself evaluated.

Required gates:

| Gate | Threshold |
|---|---:|
| Valid protocol | ≥ 90% |
| Patch application | ≥ 90% |
| Atomic task success | ≥ 75% |
| Scope compliance | 100% |
| Raw evidence coverage | 100% |
| Success vs untouched base | No regression |

These are minimum release thresholds, not claims that current weights have met
them.

## Security boundary

The verifier uses temporary workspaces, argv execution without a shell, timeouts,
resource limits where supported, environment reduction, path traversal rejection,
and planner-owned commands. This reduces risk but is not a hardened hostile-code
sandbox. Public or untrusted evaluation must run inside an OS container or VM with
network disabled and explicit CPU, memory, process, and filesystem isolation.

## Evidence that must ship

- foundation lock and immutable model revision;
- dataset and preference manifests with hashes;
- SFT/DPO run manifests;
- raw held-out outputs;
- command exit codes, stdout/stderr excerpts, and timing;
- untouched-base comparison;
- GGUF hashes and per-quantization evaluation;
- honest model and dataset cards with known failure cases.
