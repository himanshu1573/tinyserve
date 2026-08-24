"""Per-request state. A dataclass with stop logic and nothing else.

Session 7 adds a block table here. Keeping this free of engine logic is
what makes that change small.
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

    # Set by the server; the engine thread pushes tokens across with these.
    # Untyped to keep asyncio out of this module's imports.
    out_queue: Any = None
    loop: Any = None

    def append(self, token_id: int) -> None:
        self.generated.append(token_id)

    def all_tokens(self) -> list[int]:
        return self.prompt_tokens + self.generated

    def should_stop(self, eos_id: int) -> bool:
        """Check stop conditions and record why. EOS is checked first so a
        sequence whose final token is EOS reports "stop", not "length"."""
        if self.generated and self.generated[-1] == eos_id:
            self.stop_reason = "stop"
            return True
        if len(self.generated) >= self.max_tokens:
            self.stop_reason = "length"
            return True
        return False
