"""Pinned local Qwen inference with bounded, deterministic JSON tool generation."""

import re
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any

from genrecagent.llm.base import BaseLLM
from genrecagent.llm.tool_json import decode_tool_json, tool_messages


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    text: str
    input_tokens: int
    output_tokens: int
    elapsed_seconds: float
    reached_token_limit: bool


class LocalQwenLLM(BaseLLM):
    """Load only the pinned local snapshot; no provider key or service is required."""

    @classmethod
    def from_loaded_model(cls, model, tokenizer, *, device: str, model_name: str, revision: str):
        """Use an already loaded model for validation rollouts; do not reload weights."""
        import torch
        from transformers import GenerationConfig

        instance = cls.__new__(cls)
        instance.torch, instance.model, instance.tokenizer = torch, model, tokenizer
        instance.model_name, instance.revision, instance.device = model_name, revision, device
        instance.max_input_tokens, instance.max_new_tokens = 1024, 96
        instance.generation_config = GenerationConfig(
            max_new_tokens=96,
            do_sample=False,
            use_cache=True,
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        instance.last_generation = None
        instance.runtime = {
            "provider": "loaded_local_qwen",
            "model_name": model_name,
            "revision": revision,
            "device": device,
            "do_sample": False,
        }
        return instance

    def __init__(
        self,
        *,
        model_name: str,
        revision: str,
        cache_dir: Path,
        device: str = "cuda",
        dtype: str = "float16",
        max_input_tokens: int = 2048,
        max_new_tokens: int = 192,
        allow_download: bool = False,
        adapter_path: Path | None = None,
    ) -> None:
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("A full immutable model revision is required")
        if device not in {"cpu", "cuda"} or dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("Unsupported device or dtype")
        if min(max_input_tokens, max_new_tokens) < 1:
            raise ValueError("Token budgets must be positive")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self.torch = torch
        self.model_name, self.revision, self.device = model_name, revision, device
        self.max_input_tokens, self.max_new_tokens = max_input_tokens, max_new_tokens
        snapshot = cache_dir / ("models--" + model_name.replace("/", "--")) / "snapshots" / revision
        location = str(snapshot) if snapshot.is_dir() else model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            location,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=not allow_download,
            trust_remote_code=False,
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                location,
                revision=revision,
                cache_dir=cache_dir,
                local_files_only=not allow_download,
                trust_remote_code=False,
                dtype=getattr(torch, dtype),
            )
            .to(device)
            .eval()
        )
        if self.model.config._commit_hash != revision and not (
            snapshot.is_dir() and location == str(snapshot) and snapshot.name == revision
        ):
            raise RuntimeError("Loaded model revision differs from the pinned configuration")
        if adapter_path is not None:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path).eval()
        self.generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=self.model.generation_config.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        self.last_generation: GenerationRecord | None = None
        self.runtime = {
            "provider": "local_qwen",
            "model_name": model_name,
            "revision": revision,
            "device": device,
            "dtype": dtype,
            "max_input_tokens": max_input_tokens,
            "max_new_tokens": max_new_tokens,
            "do_sample": self.generation_config.do_sample,
            "torch_version": torch.__version__,
            "transformers_version": version("transformers"),
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "adapter_path": str(adapter_path) if adapter_path else None,
        }

    def _generate(self, messages: list[dict[str, str]]) -> str:
        self.last_generation = None
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        input_length = inputs["input_ids"].shape[1]
        if input_length > self.max_input_tokens:
            raise ValueError("Input exceeds the configured token budget")
        started = perf_counter()
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, generation_config=self.generation_config)
        if self.device == "cuda":
            self.torch.cuda.synchronize()
        suffix = generated[0, input_length:]
        text = self.tokenizer.decode(suffix, skip_special_tokens=True).strip()
        self.last_generation = GenerationRecord(
            text=text,
            input_tokens=int(input_length),
            output_tokens=len(suffix),
            elapsed_seconds=perf_counter() - started,
            reached_token_limit=len(suffix) >= self.max_new_tokens,
        )
        return text

    def generate(self, prompt: str) -> str:
        return self._generate([{"role": "user", "content": prompt}])

    def chat(self, messages: list[dict[str, str]]) -> str:
        return self._generate(messages)

    def tool_call(self, prompt: str, tools: list[dict[str, Any]]) -> dict[str, Any]:
        return decode_tool_json(self._generate(tool_messages(prompt, tools)))

    def generation_record(self) -> dict[str, object] | None:
        return asdict(self.last_generation) if self.last_generation else None
