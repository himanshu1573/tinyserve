"""One thread owns every MLX call.

MLX calls block. Running one inside an async request handler stalls the
event loop and every other connection with it. So the model lives on a
dedicated thread and tokens cross back to the event loop through
per-request asyncio queues via ``loop.call_soon_threadsafe``.

Message protocol on ``seq.out_queue``:

    ("text", str)          a detokenized segment
    ("error", str)         something went wrong; a "done" still follows
    ("done", finish_reason)   exactly once, always — or the handler hangs

The engine thread's loop is: drain the intake queue into the scheduler,
run one scheduler step if there is work, otherwise block on intake.
"""

import queue
import threading
import time

from tinyserve.config import EngineConfig
from tinyserve.engine.backends import make_backend
from tinyserve.engine.scheduler import Scheduler
from tinyserve.engine.sequence import Sequence, SeqStatus

_SHUTDOWN = object()


class Engine:
    def __init__(self, runner, config: EngineConfig | None = None):
        self.runner = runner
        self.config = (config or EngineConfig()).validate()
        backend = make_backend(
            self.config.kv_backend,
            runner.kv_spec,
            int(self.config.kv_budget_gb * 1024**3),
            block_size=self.config.block_size,
            prefix_caching=self.config.prefix_caching,
        )
        self.scheduler = Scheduler(
            runner, backend,
            max_batch_size=self.config.max_batch_size,
            scheduling=self.config.scheduling,
            max_prefill_per_step=self.config.max_prefill_per_step,
            on_token=self._on_token,
            on_finish=self._on_finish,
        )
        self._intake: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._detoks: dict[str, object] = {}
        self._lock = threading.Lock()

    # --- public, any thread ---------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="tinyserve-engine")
        self._thread.start()

    def stop(self) -> None:
        self._intake.put(_SHUTDOWN)
        if self._thread:
            self._thread.join(timeout=5)

    def submit(self, seq: Sequence) -> None:
        seq.t_submitted = time.perf_counter()
        self._intake.put(seq)

    def stats(self) -> dict:
        with self._lock:
            return dict(self.scheduler.stats())

    # --- engine thread below this line ----------------------------------------

    def _run(self) -> None:
        sched = self.scheduler
        while True:
            # Block only when idle; otherwise just drain whatever has arrived.
            block = not sched.has_work()
            try:
                item = self._intake.get(block=block)
            except queue.Empty:
                item = None
            while item is not None:
                if item is _SHUTDOWN:
                    return
                self._intake_one(item)
                try:
                    item = self._intake.get_nowait()
                except queue.Empty:
                    item = None

            if sched.has_work():
                try:
                    with self._lock:
                        sched.step()
                except Exception as exc:  # never let the thread die silently
                    self._fail_all(f"{type(exc).__name__}: {exc}")

    def _intake_one(self, seq: Sequence) -> None:
        self._detoks[seq.id] = self.runner.new_detokenizer()
        with self._lock:
            self.scheduler.add(seq)

    def _on_token(self, seq: Sequence, token: int) -> None:
        if seq.t_first_token == 0.0:
            seq.t_first_token = time.perf_counter()
        if token in self.runner.eos_ids:
            return
        detok = self._detoks[seq.id]
        detok.add_token(token)
        segment = detok.last_segment
        if segment:
            self._emit(seq, ("text", segment))

    def _on_finish(self, seq: Sequence) -> None:
        seq.t_finished = time.perf_counter()
        detok = self._detoks.pop(seq.id, None)
        if detok is not None:
            detok.finalize()
            if detok.last_segment:
                self._emit(seq, ("text", detok.last_segment))
        if seq.error:
            self._emit(seq, ("error", seq.error))
        self._emit(seq, ("done", seq.stop_reason or "stop"))

    def _fail_all(self, message: str) -> None:
        """A forward blew up. Every in-flight request learns why, and the
        scheduler starts over empty rather than retrying a broken batch."""
        sched = self.scheduler
        victims = list({id(s): s for s in [*sched.running, *sched.waiting]}.values())
        sched.running.clear()
        sched.waiting.clear()
        for seq in victims:
            try:
                sched.backend.free(seq)
            except Exception:
                pass
            seq.error = message
            seq.status = SeqStatus.FINISHED
            seq.stop_reason = "error"
            self._on_finish(seq)

    @staticmethod
    def _emit(seq: Sequence, message) -> None:
        """call_soon_threadsafe is the only safe way to touch an
        asyncio.Queue from a non-async thread."""
        if seq.out_queue is None or seq.loop is None:
            return
        seq.loop.call_soon_threadsafe(seq.out_queue.put_nowait, message)
