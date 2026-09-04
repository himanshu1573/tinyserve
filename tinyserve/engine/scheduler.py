"""Continuous batching — the heart of the engine.

One ``step()`` is the loop from the Anyscale post, written down:

    admit  -> prefill each newly admitted sequence (B=1), emit its first token
    decode -> one batched forward for everything RUNNING, emit one token each
    evict  -> free finished sequences so the next step can admit more

In "static" mode the admit phase waits until the batch has drained, which
is the Session 4-5 engine; "continuous" admits into the running batch
every step. Same code, one condition — that is the whole difference, and
M5 measures what it buys.

The scheduler owns no MLX state and no I/O. It reports through two
callbacks, ``on_token(seq, token_id)`` and ``on_finish(seq)``, which is
what makes it testable with a fake runner in milliseconds.
"""

from collections import deque
from typing import Callable

from tinyserve.engine.sampler import sample, sample_batch
from tinyserve.engine.sequence import Sequence, SeqStatus


class Scheduler:
    def __init__(self, runner, backend, *, max_batch_size: int = 8,
                 scheduling: str = "continuous", max_prefill_per_step: int = 4,
                 on_token: Callable[[Sequence, int], None] | None = None,
                 on_finish: Callable[[Sequence], None] | None = None):
        self.runner = runner
        self.backend = backend
        self.max_batch_size = max_batch_size
        self.scheduling = scheduling
        self.max_prefill_per_step = max_prefill_per_step
        self.on_token = on_token or (lambda seq, tok: None)
        self.on_finish = on_finish or (lambda seq: None)

        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.num_steps = 0
        self.num_prefills = 0
        self.num_preemptions = 0

    # --- intake -------------------------------------------------------------

    def add(self, seq: Sequence) -> None:
        # A prompt that can never fit must fail now, not wait forever.
        if seq.num_tokens + 1 > self.backend.capacity_tokens:
            seq.error = (f"prompt of {seq.num_tokens} tokens exceeds the KV budget "
                         f"of {self.backend.capacity_tokens} tokens")
            self._finish(seq, "error")
            return
        seq.status = SeqStatus.WAITING
        self.waiting.append(seq)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    # --- one step -----------------------------------------------------------

    def step(self) -> None:
        self.num_steps += 1
        self._admit()
        self._decode()

    def _admit(self) -> None:
        if self.scheduling == "static" and self.running:
            return
        admitted = 0
        while (self.waiting and len(self.running) < self.max_batch_size
               and admitted < self.max_prefill_per_step):
            seq = self.waiting[0]
            if not self.backend.can_admit(seq):
                break
            self._prefill(seq)
            admitted += 1

    def _prefill(self, seq: Sequence) -> None:
        """Prefill waiting[0]. It leaves the waiting queue only once the
        forward has succeeded, so a failure inside the runner still leaves
        the sequence somewhere the engine can find it and report on it."""
        self.backend.admit(seq)
        caches = self.backend.prefill_caches(seq)
        tokens = seq.all_tokens()[seq.num_cached_tokens:]
        logits = self.runner.prefill(tokens, caches)
        self.backend.after_prefill(seq, caches)
        self.num_prefills += 1

        self.waiting.popleft()
        seq.status = SeqStatus.RUNNING
        self.running.append(seq)
        token = sample(logits, seq.temperature, seq.top_p)
        self._emit(seq, token)

    def _decode(self) -> None:
        if not self.running:
            return
        # Every running sequence needs a slot for the token it is about to
        # feed in. If the pool cannot provide one, the newest sequence gives
        # its memory back and waits; it re-prefills later.
        while self.running:
            if all(self.backend.may_append(s) for s in self.running):
                break
            self._preempt(self.running[-1])
        if not self.running:
            return

        batch = list(self.running)
        caches = self.backend.decode_caches(batch)
        logits = self.runner.decode_batch([s.last_token for s in batch], caches)
        tokens = sample_batch(logits, batch)
        for seq, token in zip(batch, tokens):
            self._emit(seq, token)

    # --- bookkeeping ---------------------------------------------------------

    def _emit(self, seq: Sequence, token: int) -> None:
        seq.append(token)
        self.on_token(seq, token)
        if seq.should_stop(self.runner.eos_ids):
            self.running.remove(seq)
            self.backend.free(seq)
            self._finish(seq, seq.stop_reason)

    def _finish(self, seq: Sequence, reason: str) -> None:
        seq.status = SeqStatus.FINISHED
        seq.stop_reason = reason
        self.on_finish(seq)

    def _preempt(self, seq: Sequence) -> None:
        self.running.remove(seq)
        self.backend.free(seq)
        seq.status = SeqStatus.WAITING
        seq.preemptions += 1
        self.num_preemptions += 1
        self.waiting.appendleft(seq)

    def stats(self) -> dict:
        return {
            "waiting": len(self.waiting),
            "running": len(self.running),
            "max_batch_size": self.max_batch_size,
            "scheduling": self.scheduling,
            "steps": self.num_steps,
            "prefills": self.num_prefills,
            "preemptions": self.num_preemptions,
            **self.backend.stats(),
        }
