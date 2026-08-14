from __future__ import annotations

import gc
import logging
import os

from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)

from niftsy.exceptions import NiftsyError
from niftsy.llm.base import UsageTracker
from niftsy.llm.gpu import select_free_gpu

LOGGER = logging.getLogger(__name__)

_DEFAULT_STOP_SEQUENCES = [
    "[INST]", "[/INST]", "</s>", "<|im_start|>", "<|im_end|>",
    "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|end_of_text|>",
]

# transformers (which vLLM's model loading goes through) frequently re-wraps
# these Hugging Face exceptions as a plain OSError with a distinctive message
# before they reach us -- so detection needs both isinstance checks (for the
# un-wrapped case) and message-substring matching (for the wrapped case).
_GATED_MESSAGE_MARKERS = ("gated repo", "authorized list", "awaiting a review")
_NOT_FOUND_MESSAGE_MARKERS = (
    "not a valid model identifier",
    "repository not found",
    "does not appear to have a file named",
    "404 client error",
)
_NETWORK_MESSAGE_MARKERS = (
    "couldn't connect",
    "could not connect",
    "connection error",
    "connectionerror",
    "max retries exceeded",
    "name or service not known",
    "connection refused",
    "read timed out",
    "connection timed out",
    "failed to establish a new connection",
)


def _exception_chain(exc, max_depth=5):
    """Walk exc, its __cause__, its __cause__'s __cause__, etc. (falling
    back to __context__ when there's no explicit `from e`), bounded so a
    pathological cycle can't loop forever."""
    seen: set[int] = set()
    current = exc
    for _ in range(max_depth):
        if current is None or id(current) in seen:
            return
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _translate_model_load_error(exc: Exception, model_name: str) -> NiftsyError | None:
    """Best-effort translation of a raw model-load failure into an
    actionable NiftsyError. Returns None if nothing matches (caller then
    decides what to do with a genuinely unclassified exception)."""
    chain = list(_exception_chain(exc))
    combined_message = " ".join(str(e) for e in chain).lower()

    # GatedRepoError is a subclass of RepositoryNotFoundError, so this check
    # must run first.
    if any(isinstance(e, GatedRepoError) for e in chain) or any(
        marker in combined_message for marker in _GATED_MESSAGE_MARKERS
    ):
        return NiftsyError(
            f"Model '{model_name}' requires Hugging Face authentication (a gated/"
            "license-restricted repo). Request access at "
            f"https://huggingface.co/{model_name} and set HUGGINGFACE_HUB_TOKEN in your .env."
        )

    if any(isinstance(e, RepositoryNotFoundError) for e in chain) or any(
        marker in combined_message for marker in _NOT_FOUND_MESSAGE_MARKERS
    ):
        return NiftsyError(
            f"Model '{model_name}' was not found on Hugging Face Hub. "
            "Check the model name for typos."
        )

    if any(marker in combined_message for marker in _NETWORK_MESSAGE_MARKERS):
        return NiftsyError(
            f"Could not download model weights for '{model_name}' (network error). "
            "Check your internet connection and try again."
        )

    return None


class LocalVLLMBackend:
    def __init__(
        self,
        model: str,
        gpu_memory_utilization: float = 0.80,
        max_model_len: int = 4096,
        enforce_eager: bool = False,
        gpu_index: int | None = None,
    ) -> None:
        self.model_name = model
        self.provider = "local"
        self.gpu_util = gpu_memory_utilization
        self.max_len = max_model_len
        self.enforce_eager = enforce_eager
        self.usage = UsageTracker()

        if gpu_index is not None:
            selected_index = int(gpu_index)
            LOGGER.info("Using forced GPU %s (gpu_index override)", selected_index)
        else:
            selected_index = select_free_gpu(self.gpu_util).index

        os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_index)

        LOGGER.info("Initializing local vLLM backend on GPU %s", selected_index)

        try:
            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=1,
                gpu_memory_utilization=self.gpu_util,
                max_model_len=self.max_len,
                trust_remote_code=True,
                enforce_eager=self.enforce_eager,
            )
        except Exception as exc:
            self.cleanup()
            message = str(exc).lower()
            if "out of memory" in message or ("cuda" in message and "memory" in message):
                raise NiftsyError(
                    "Model does not fit in one GPU. Retry with another model."
                ) from exc
            translated = _translate_model_load_error(exc, self.model_name)
            if translated is not None:
                raise translated from exc
            raise NiftsyError(
                f"Failed to load local model '{self.model_name}': "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def cleanup(self) -> None:
        """Forcefully clears GPU memory and destroys distributed processes."""
        LOGGER.info("Cleaning up GPU resources...")
        try:
            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception:
            pass

        if hasattr(self, "llm"):
            del self.llm

        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        LOGGER.info("Cleanup complete.")

    def generate_batch(self, prompts: list[str], config: dict | None = None) -> list[str]:
        config = config or {}

        stop_sequences = config.get("stop_sequences")
        if stop_sequences is None:
            stop_sequences = _DEFAULT_STOP_SEQUENCES

        sampling_params = SamplingParams(
            temperature=config.get("temperature", 0.7),
            top_p=config.get("top_p", 0.95),
            max_tokens=config.get("max_tokens", 512),
            repetition_penalty=config.get("repetition_penalty", 1.1),
            stop=stop_sequences,
        )

        # Apply the model's native chat template so instruction-tuned models
        # receive system/user/assistant roles instead of a raw text string.
        # This eliminates preambles ("Here goes:", "Sure, here's...", etc.)
        # that appear when the model tries to reason about a raw prompt.
        tokenizer = self.llm.get_tokenizer()
        if hasattr(tokenizer, "apply_chat_template"):
            system_msg = (
                "Output ONLY what is requested. "
                "No preamble, no explanation, no meta-commentary. "
                "Your first token is the first token of the actual response."
            )
            formatted: list[str] = []
            for prompt in prompts:
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ]
                formatted.append(
                    tokenizer.apply_chat_template(
                        messages,  # type: ignore[arg-type]
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
            prompts = formatted

        outputs = self.llm.generate(prompts, sampling_params)
        prompt_tokens = 0
        completion_tokens = 0
        for output in outputs:
            prompt_tokens += len(getattr(output, "prompt_token_ids", []) or [])
            first_output = output.outputs[0] if output.outputs else None
            completion_tokens += len(getattr(first_output, "token_ids", []) or [])
        self.usage.record(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            requests=len(prompts),
        )
        return [o.outputs[0].text.strip() for o in outputs]
