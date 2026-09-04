"""The only module that talks to MLX model machinery.

mlx_lm gives us two things and nothing more: loaded weights and a
tokenizer. The decode loop below is ours, because Session 4's batched
forward needs this exact call in hand, and because a baseline measured
through mlx_lm's own loop would not be a measurement of eightserve.
"""

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

DEFAULT_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"


class Runner:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.eos_id = tokenizer.eos_token_id

    @classmethod
    def load(cls, model_id: str = DEFAULT_MODEL) -> "Runner":
        model, tokenizer = load(model_id)
        return cls(model, tokenizer)

    def format_prompt(self, instruction: str) -> str:
        """Apply the model's chat template.

        Qwen2.5-Instruct is instruction-tuned; feeding it a bare
        instruction produces noticeably worse continuations. Both the CLI
        and the benchmark client call this, so the tokens the model sees
        are identical across every measurement path.
        """
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def new_cache(self) -> list:
        """A fresh per-layer KV cache. Session 7 replaces this with our own
        paged allocator; the signature is chosen so that swap is local."""
        return make_prompt_cache(self.model)

    def prefill(self, token_ids: list[int], cache) -> mx.array:
        """One forward over the whole prompt. Returns 1-D logits for the
        next position. Every prompt token is processed in parallel here —
        this is the compute-bound phase, and the source of TTFT."""
        logits = self.model(mx.array([token_ids]), cache=cache)
        out = logits[0, -1, :]
        mx.eval(out)
        return out

    def decode_step(self, token_id: int, cache) -> mx.array:
        """One token forward. The cache is mutated in place.

        This single call re-reads every weight in the model. That is the
        whole reason decode is memory-bound and why batching wins: the
        read is shared, the extra math is nearly free.
        """
        logits = self.model(mx.array([[token_id]]), cache=cache)
        out = logits[0, -1, :]
        mx.eval(out)
        return out

    @staticmethod
    def peak_memory_gb() -> float:
        """Peak bytes MLX has held, in GB. psutil's RSS misses unified
        (Metal) memory almost entirely, so this is the number that describes
        the 8 GB budget."""
        return mx.get_peak_memory() / 1024**3
