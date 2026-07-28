#!/usr/bin/env python3
"""Merge a Nova adapter and export reproducible GGUF quantisations for Ollama."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(argv: list[str]) -> None:
    print("+", " ".join(argv))
    subprocess.run(argv, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Nova V12 to GGUF")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--convert-script",
        type=Path,
        required=True,
        help="llama.cpp convert_hf_to_gguf.py",
    )
    parser.add_argument("--quantize-bin", type=Path, required=True)
    parser.add_argument(
        "--quant",
        action="append",
        choices=["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"],
        default=None,
    )
    args = parser.parse_args()

    provenance = args.adapter / "nova_dpo_run.json"
    if not provenance.is_file():
        provenance = args.adapter / "nova_sft_run.json"
    if not provenance.is_file():
        parser.exit(2, "adapter lacks Nova training provenance\n")
    if not args.convert_script.is_file():
        parser.exit(2, f"missing llama.cpp converter: {args.convert_script}\n")
    if not args.quantize_bin.is_file():
        parser.exit(2, f"missing llama.cpp quantizer: {args.quantize_bin}\n")

    try:
        import torch
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer
    except ImportError as exc:
        parser.exit(2, f"missing export dependency: {exc}\n")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged = args.output_dir / "merged-hf"
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.adapter,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model = model.merge_and_unload()
    model.save_pretrained(merged, safe_serialization=True, max_shard_size="4GB")
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(merged)

    f16 = args.output_dir / "nova-v12-f16.gguf"
    run(
        [
            "python",
            str(args.convert_script),
            str(merged),
            "--outfile",
            str(f16),
            "--outtype",
            "f16",
        ]
    )
    quantisations = args.quant or ["Q8_0", "Q5_K_M", "Q4_K_M"]
    artifacts: list[dict[str, object]] = []
    for quant in quantisations:
        destination = args.output_dir / f"nova-v12-{quant.lower()}.gguf"
        run([str(args.quantize_bin), str(f16), str(destination), quant])
        if not destination.is_file() or destination.stat().st_size == 0:
            parser.exit(2, f"quantisation did not produce {destination}\n")
        artifacts.append(
            {
                "path": destination.name,
                "quantisation": quant,
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        )

    default = next(item for item in artifacts if item["quantisation"] == "Q4_K_M")
    modelfile = args.output_dir / "Modelfile"
    modelfile.write_text(
        "\n".join(
            [
                f"FROM ./{default['path']}",
                "PARAMETER temperature 0",
                "PARAMETER top_p 1",
                "PARAMETER num_ctx 8192",
                'SYSTEM """You are Nova V12, Amaura Labs\' local atomic patch executor. '
                'Return only the Nova JSON patch or escalation protocol."""',
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "nova.export.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "adapter": str(args.adapter),
        "training_provenance_sha256": sha256_file(provenance),
        "llama_cpp": {
            "convert_script": str(args.convert_script),
            "quantize_bin": str(args.quantize_bin),
        },
        "artifacts": artifacts,
        "modelfile_sha256": sha256_file(modelfile),
        "release_status": "candidate_requires_quantized_evaluation",
    }
    (args.output_dir / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
