"""One thread owns every MLX call.

MLX calls block. Running one inside an async request handler stalls the
event loop and every other connection with it. So the model lives on a
dedicated thread and tokens cross back to the event loop through
per-request asyncio queues.

Session 2 runs one sequence at a time. Session 6 replaces the body of
_run() with a scheduler that admits and evicts sequences every step —
this file's threading contract is what that change plugs into, and it is
the reason the contract exists before the batching does.
"""

import queue
import threading

from eightserve.engine.sampler import sample
from eightserve.engine.sequence import Sequence, SeqStatus

_SHUTDOWN = object()


class Engine:
    def __init__(self, runner):
        self.runner = runner
        self._intake: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="eightserve-engine")
        self._thread.start()

    def stop(self) -> None:
        self._intake.put(_SHUTDOWN)
        if self._thread:
            self._thread.join(timeout=5)

    def submit(self, seq: Sequence) -> None:
        self._intake.put(seq)

    # --- engine thread below this line ---

    def _run(self) -> None:
        while True:
            item = self._intake.get()
            if item is _SHUTDOWN:
                return
            try:
                self._generate(item)
            except Exception as exc:  # never let the thread die silently
                self._emit(item, ("error", f"{type(exc).__name__}: {exc}"))
                item.status = SeqStatus.FINISHED
                self._emit(item, ("done", "error"))

    def _generate(self, seq: Sequence) -> None:
        seq.status = SeqStatus.RUNNING
        runner = self.runner

        detok = runner.tokenizer.detokenizer
        detok.reset()
        cache = runner.new_cache()

        logits = runner.prefill(seq.prompt_tokens, cache)

        while True:
            token = sample(logits, temperature=seq.temperature, top_p=seq.top_p)
            seq.append(token)

            # Emit before checking stop conditions, but never emit EOS.
            # Checking first would silently swallow the final token whenever
            # a sequence ends by hitting max_tokens.
            if token != runner.eos_id:
                detok.add_token(token)
                segment = detok.last_segment
                if segment:
                    self._emit(seq, ("text", segment))

            if seq.should_stop(runner.eos_id):
                break

            logits = runner.decode_step(token, cache)

        detok.finalize()
        if detok.last_segment:
            self._emit(seq, ("text", detok.last_segment))

        seq.status = SeqStatus.FINISHED
        self._emit(seq, ("done", seq.stop_reason or "stop"))

    @staticmethod
    def _emit(seq: Sequence, message) -> None:
        """Hand a message to the event loop that owns this sequence's queue.
        call_soon_threadsafe is the only safe way to touch an asyncio.Queue
        from a non-async thread."""
        seq.loop.call_soon_threadsafe(seq.out_queue.put_nowait, message)
