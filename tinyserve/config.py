"""Engine configuration — one dataclass, no magic.

Every knob that changes a measurement lives here, so a benchmark row can
be labelled with the exact configuration that produced it.
"""

from dataclasses import dataclass, asdict

DEFAULT_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"


@dataclass
class EngineConfig:
    model: str = DEFAULT_MODEL

    # Scheduler ---------------------------------------------------------
    # 1 reproduces the serial engine of sessions 1-3 exactly.
    max_batch_size: int = 8
    # "continuous": admit and evict every step (session 6).
    # "static": admit a batch, run it to completion, admit the next (sessions 4-5).
    scheduling: str = "continuous"
    # How many waiting sequences may be prefilled in a single step before
    # the running batch gets its decode step. Bounds TTFT for the running
    # users when a burst of new requests arrives.
    max_prefill_per_step: int = 4

    # KV cache ----------------------------------------------------------
    # "paged": block allocator + slot pool (sessions 7-9).
    # "padded": one left-padded (B, H, L, D) tensor per layer (sessions 4-6).
    kv_backend: str = "paged"
    # Total bytes the KV cache may occupy, all layers, K and V. On an 8 GB
    # machine with a ~1 GB model and a browser open this is the number that
    # decides how many users fit.
    kv_budget_gb: float = 0.5
    block_size: int = 16
    prefix_caching: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> "EngineConfig":
        if self.scheduling not in ("continuous", "static"):
            raise ValueError(f"scheduling must be continuous|static, got {self.scheduling!r}")
        if self.kv_backend not in ("paged", "padded"):
            raise ValueError(f"kv_backend must be paged|padded, got {self.kv_backend!r}")
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if self.block_size < 1:
            raise ValueError("block_size must be >= 1")
        return self
