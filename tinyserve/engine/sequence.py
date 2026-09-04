"""Per-request state. A dataclass with stop logic and nothing else.

Everything the scheduler, the block manager and the server need to know
about one request lives here, and none of the logic does. Keeping this
module free of engine behaviour is what keeps every other module small.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SeqStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class Sequence:
    id: str
    prompt_tokens: list[int]
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0

    generated: list[int] = field(default_factory=list)
    status: SeqStatus = SeqStatus.WAITING
    stop_reason: str | None = None
    error: str | None = None

    # Paged KV state (sessions 7-9). Logical block i of this sequence lives
    # in physical block block_table[i]. num_cached_tokens counts leading
    # tokens whose KV was found in shared prefix blocks and need no prefill.
    block_table: list[int] = field(default_factory=list)
    num_cached_tokens: int = 0
    preemptions: int = 0

    # Set by the server; the engine thread pushes messages across with
    # these. Untyped to keep asyncio out of this module's imports.
    out_queue: Any = None
    loop: Any = None

    # Timing, filled in by the engine (perf_counter seconds).
    t_submitted: float = 0.0
    t_first_token: float = 0.0
    t_finished: float = 0.0

    # --- token bookkeeping -------------------------------------------------

    def append(self, token_id: int) -> None:
        self.generated.append(token_id)

    def all_tokens(self) -> list[int]:
        return self.prompt_tokens + self.generated

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_tokens) + len(self.generated)

    @property
    def last_token(self) -> int:
        return self.generated[-1] if self.generated else self.prompt_tokens[-1]

    def block_tokens(self, i: int, block_size: int) -> list[int]:
        """Tokens that occupy logical block i (may be partial for the last)."""
        return self.all_tokens()[i * block_size : (i + 1) * block_size]

    def num_blocks(self, block_size: int) -> int:
        return (self.num_tokens + block_size - 1) // block_size

    # --- stop conditions ---------------------------------------------------

    def should_stop(self, eos_ids) -> bool:
        """Check stop conditions and record why. EOS is checked first so a
        sequence whose final token is EOS reports "stop", not "length"."""
        if isinstance(eos_ids, int):
            eos_ids = {eos_ids}
        if self.generated and self.generated[-1] in eos_ids:
            self.stop_reason = "stop"
            return True
        if len(self.generated) >= self.max_tokens:
            self.stop_reason = "length"
            return True
        return False

    def reset_kv_state(self) -> None:
        """Forget where the KV lives. Called on preemption: the tokens stay,
        the memory goes, and the next admission re-prefills them."""
        self.block_table = []
        self.num_cached_tokens = 0
