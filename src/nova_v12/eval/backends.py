from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from nova_v12.schemas import EvalTask


@dataclass(slots=True)
class BackendOutput:
    text: str
    latency_seconds: float
    tokens_generated: int
    tokens_per_second: float
    revision: str | None = None


class GenerationBackend(Protocol):
    name: str

    def generate(self, prompt: str, *, max_tokens: int, temperature: float) -> BackendOutput: ...


class OllamaBackend:
    name = "ollama"

    def __init__(self, model: str, base_url: str | None = None) -> None:
        self.model = model
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip(
            "/"
        )

    def generate(self, prompt: str, *, max_tokens: int, temperature: float) -> BackendOutput:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens, "seed": 42},
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=600) as response:
            body = json.loads(response.read().decode("utf-8"))
        latency = time.perf_counter() - started
        tokens = int(body.get("eval_count", 0))
        duration_ns = int(body.get("eval_duration", 0))
        tps = (
            tokens / (duration_ns / 1e9)
            if duration_ns > 0
            else (tokens / latency if latency else 0.0)
        )
        return BackendOutput(body.get("response", ""), latency, tokens, tps, body.get("model"))


class TransformersBackend:
    name = "transformers"

    def __init__(
        self, model_id: str, *, trust_remote_code: bool = False, revision: str | None = None
    ) -> None:
        self.model_id = model_id
        self.trust_remote_code = trust_remote_code
        self.revision = revision
        self.model = None
        self.tokenizer = None

    def _load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
        )
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.model.eval()

    def _format(self, prompt: str) -> str:
        self._load()
        assert self.tokenizer is not None
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def _generate_formatted(
        self, formatted: str, *, max_tokens: int, temperature: float
    ) -> BackendOutput:
        import torch

        self._load()
        assert self.tokenizer is not None and self.model is not None
        inputs = self.tokenizer(formatted, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        input_length = inputs["input_ids"].shape[1]
        started = time.perf_counter()
        torch.manual_seed(42)
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                pad_token_id=self.tokenizer.eos_token_id,
            )
        latency = time.perf_counter() - started
        tokens = int(generated.shape[1] - input_length)
        text = self.tokenizer.decode(generated[0, input_length:], skip_special_tokens=True)
        return BackendOutput(
            text, latency, tokens, tokens / latency if latency else 0.0, self.revision
        )

    def generate(self, prompt: str, *, max_tokens: int, temperature: float) -> BackendOutput:
        return self._generate_formatted(
            self._format(prompt), max_tokens=max_tokens, temperature=temperature
        )

    def generate_for_task(
        self, task: "EvalTask", prompt: str, *, max_tokens: int, temperature: float
    ) -> BackendOutput:
        self._load()
        assert self.tokenizer is not None
        if task.category == "fim":
            vocabulary = self.tokenizer.get_vocab()
            candidates = [
                ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"),
                ("<fim_prefix>", "<fim_suffix>", "<fim_middle>"),
            ]
            for prefix_token, suffix_token, middle_token in candidates:
                if all(token in vocabulary for token in (prefix_token, suffix_token, middle_token)):
                    formatted = (
                        prefix_token + task.prefix + suffix_token + task.suffix + middle_token
                    )
                    return self._generate_formatted(
                        formatted, max_tokens=max_tokens, temperature=temperature
                    )
        return self.generate(prompt, max_tokens=max_tokens, temperature=temperature)
