"""A fake runner for the fast tests.

It is not a mock of MLX. It runs a *fake forward* that calls every cache's
``make_mask`` and ``update_and_fetch`` with correctly shaped arrays, so the
padded and paged plumbing is exercised for real; only the transformer is
replaced. The "model" is: next token = last token + 1, and EOS whenever
that would be a multiple of 10. Different prompts therefore produce
different continuations, which is how row mix-ups get caught.
"""

import mlx.core as mx

from tinyserve.engine.backends import KVSpec
from tinyserve.engine.kv_cache import KVCache

EOS = 999
VOCAB = 1000


class FakeDetok:
    """Same contract as mlx-lm's StreamingDetokenizer: ``last_segment`` is
    the text produced since it was last read, so reading it twice gives
    the segment once and then an empty string."""

    def __init__(self):
        self.text = ""
        self.offset = 0

    def reset(self):
        self.text, self.offset = "", 0

    def add_token(self, t):
        self.text += f"<{t}>"

    def finalize(self):
        pass

    @property
    def last_segment(self):
        segment = self.text[self.offset:]
        self.offset = len(self.text)
        return segment


def expected_continuation(prompt: list[int], max_tokens: int) -> tuple[list[int], str]:
    """What the fake model generates for a prompt (EOS excluded), plus the
    finish reason."""
    out = []
    t = prompt[-1]
    while len(out) < max_tokens:
        t = t + 1
        if t % 10 == 0:
            return out, "stop"
        out.append(t)
    return out, "length"


class FakeRunner:
    eos_ids = {EOS}
    eos_id = EOS
    kv_spec = KVSpec(num_layers=2, n_kv_heads=1, head_dim=4)

    def __init__(self):
        self.prefills = 0
        self.decode_calls = 0
        self.batch_sizes = []

    # --- tokenizer surface ---------------------------------------------------
    def encode(self, text):
        return [1, 2, 3]

    def format_prompt(self, text, system=None):
        return text

    def format_chat(self, messages):
        return messages[-1]["content"]

    def new_detokenizer(self):
        return FakeDetok()

    def new_cache(self):
        return [KVCache() for _ in range(self.kv_spec.num_layers)]

    def memory_stats(self):
        return {"mlx_active_gb": 0.0, "mlx_peak_gb": 0.0}

    def reset_peak_memory(self):
        pass

    # --- the fake forward ----------------------------------------------------
    def _forward(self, ids: mx.array, caches):
        B, N = ids.shape
        H, D = self.kv_spec.n_kv_heads, self.kv_spec.head_dim
        for c in caches:
            if hasattr(c, "make_mask"):
                c.make_mask(N)
            k, v = c.update_and_fetch(mx.zeros((B, H, N, D)), mx.zeros((B, H, N, D)))
            mx.eval(k, v)
        last = ids[:, -1].tolist()
        rows = []
        for t in last:
            nxt = t + 1
            if nxt % 10 == 0:
                nxt = EOS
            row = [0.0] * VOCAB
            row[nxt] = 10.0
            rows.append(row)
        return mx.array(rows)

    def prefill(self, token_ids, caches):
        self.prefills += 1
        return self._forward(mx.array([token_ids]), caches)[0]

    def decode_step(self, token_id, caches):
        return self._forward(mx.array([[token_id]]), caches)[0]

    def decode_batch(self, token_ids, caches):
        self.decode_calls += 1
        self.batch_sizes.append(len(token_ids))
        return self._forward(mx.array(token_ids)[:, None], caches)


class BrokenRunner(FakeRunner):
    def prefill(self, token_ids, caches):
        raise RuntimeError("boom")
