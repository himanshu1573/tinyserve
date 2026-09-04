"""The only module that talks to the MLX *model*.

mlx_lm gives us two things and nothing more: loaded weights and a
tokenizer. The forward calls below are ours, because the batched and
paged paths need this exact call in hand, and because a baseline measured
through mlx_lm's own loop would not be a measurement of tinyserve.

Every public method that returns logits calls mx.eval() first. MLX is
lazy; a timer around a call that does not eval measures graph building,
not work.
"""

import mlx.core as mx
from mlx_lm import load

from tinyserve.config import DEFAULT_MODEL
from tinyserve.engine.backends import KVSpec
from tinyserve.engine.kv_cache import KVCache


class Runner:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        eos = set(getattr(tokenizer, "eos_token_ids", None) or [])
        if tokenizer.eos_token_id is not None:
            eos.add(tokenizer.eos_token_id)
        self.eos_ids: set[int] = eos
        self.eos_id: int = tokenizer.eos_token_id

        args = model.args
        self.kv_spec = KVSpec(
            num_layers=args.num_hidden_layers,
            n_kv_heads=args.num_key_value_heads,
            head_dim=args.hidden_size // args.num_attention_heads,
            itemsize=model.model.norm.weight.dtype.size,
        )

    @classmethod
    def load(cls, model_id: str = DEFAULT_MODEL) -> "Runner":
        model, tokenizer = load(model_id)
        return cls(model, tokenizer)

    # --- text <-> tokens ---------------------------------------------------

    def format_prompt(self, instruction: str, system: str | None = None) -> str:
        """Apply the model's chat template. Qwen2.5-Instruct is
        instruction-tuned; a bare instruction produces worse text. Every
        measurement path calls this so the model sees identical tokens."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": instruction})
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def format_chat(self, messages: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def new_detokenizer(self):
        return self.tokenizer.detokenizer

    # --- memory ------------------------------------------------------------

    @staticmethod
    def memory_stats() -> dict:
        """MLX's own counters. psutil's RSS does not see Metal buffers, so
        this is the number that actually describes the 8 GB budget."""
        return {
            "mlx_active_gb": mx.get_active_memory() / 1024**3,
            "mlx_peak_gb": mx.get_peak_memory() / 1024**3,
        }

    @staticmethod
    def reset_peak_memory() -> None:
        mx.reset_peak_memory()

    # --- the forward, three ways ----------------------------------------------

    def new_cache(self) -> list[KVCache]:
        """A fresh contiguous per-layer cache for the single-sequence path."""
        return [KVCache() for _ in range(self.kv_spec.num_layers)]

    def forward(self, ids: mx.array, caches: list) -> mx.array:
        """(B, N) token ids -> (B, N, vocab) logits, lazily. Prefill and
        decode are this same call with a different N; nothing else differs."""
        return self.model(ids, cache=caches)

    def prefill(self, token_ids: list[int], caches: list) -> mx.array:
        """One forward over a whole prompt (B=1). Returns 1-D logits for the
        next position. This is the compute-bound phase and the source of
        TTFT. Works with any per-layer cache, contiguous or paged."""
        if not token_ids:
            raise ValueError("prefill needs at least one token")
        out = self.forward(mx.array([token_ids]), caches)[0, -1, :]
        mx.eval(out)
        return out

    def decode_step(self, token_id: int, caches: list) -> mx.array:
        """One token, one sequence. The reference the batched path is
        diffed against. This single call re-reads every weight in the
        model — the reason decode is memory-bound and batching wins."""
        out = self.forward(mx.array([[token_id]]), caches)[0, -1, :]
        mx.eval(out)
        return out

    def decode_batch(self, token_ids: list[int], caches: list) -> mx.array:
        """One token for each of B sequences in a single forward. Same
        weight read as decode_step, B times the useful work. Returns
        (B, vocab) logits."""
        ids = mx.array(token_ids)[:, None]
        out = self.forward(ids, caches)[:, -1, :]
        mx.eval(out)
        return out
