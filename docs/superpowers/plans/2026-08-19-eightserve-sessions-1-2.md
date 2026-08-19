# eightserve Sessions 1–2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A decode loop we own end-to-end, exposed over HTTP with SSE streaming, with measurements M1 and M2 recorded.

**Architecture:** `mlx_lm.load()` supplies weights and tokenizer; everything after that is ours — cache construction, prefill, decode step, sampling. One dedicated engine thread owns all MLX calls; FastAPI handlers never touch the model and receive tokens through per-request `asyncio.Queue`s.

**Tech Stack:** Python 3.12 (uv venv), mlx-lm, FastAPI, uvicorn, psutil, pytest.

**Spec:** `docs/superpowers/specs/2026-08-19-eightserve-design.md`

## Global Constraints

- Python **3.12** via `uv venv --python 3.12`. System Python is 3.14.3 and has no mlx-lm wheel.
- Model: **`mlx-community/Qwen2.5-1.5B-Instruct-4bit`**, and no other. Changing the model invalidates every recorded measurement.
- Time around **`mx.eval()`**, never around the Python call. MLX is lazy; a timer that does not bracket an eval measures nothing.
- The frozen prompt set is fixed in Task 5 and **never changes**.
- Every benchmark: **3 runs after cooldown, report the median.** The M1 Air is fanless.
- No MLX import outside `engine/runner.py` except in tests that exercise `sampler`.
- Request handlers **never** call the model. This is a correctness rule, not style.
- Every task that produces a number appends to `MEASUREMENTS.md` with the exact command and the date `2026-08-19`.

---

### Task 1: Environment, skeleton, capture files

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `MEASUREMENTS.md`, `SURPRISES.md`, `LEARNING-LOG.md`
- Create: `eightserve/__init__.py`, `eightserve/engine/__init__.py`, `eightserve/server/__init__.py`, `tests/__init__.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: nothing
- Produces: a working venv at `.venv`, `pytest` runnable, `import mlx_lm` proven

- [ ] **Step 1: Create the venv**

```bash
cd /Users/himanshup/AI_inference
uv venv --python 3.12
```

Expected: `Creating virtual environment at: .venv`. If uv reports that 3.12 is unavailable, it will download it — let it.

- [ ] **Step 2: Write pyproject.toml**

```toml
[project]
name = "eightserve"
version = "0.1.0"
description = "A readable LLM serving engine for an 8 GB M1"
requires-python = ">=3.12,<3.13"
dependencies = [
    "mlx-lm>=0.20",
    "fastapi>=0.115",
    "uvicorn>=0.32",
    "psutil>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-asyncio>=0.24", "httpx>=0.27"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
markers = ["slow: loads the real model (deselect with '-m \"not slow\"')"]
asyncio_mode = "auto"
```

- [ ] **Step 3: Install**

```bash
uv pip install -e ".[dev]"
```

Expected: mlx, mlx-lm, fastapi and friends resolve. If mlx fails to build, the Python version is wrong — check `.venv/bin/python -V` reads 3.12.x before debugging anything else.

- [ ] **Step 4: Prove the import and record the real API surface**

This step exists because mlx-lm's API moves between versions, and later tasks name specific symbols. Verify them now rather than discovering a rename mid-loop.

```bash
.venv/bin/python - <<'EOF'
import mlx.core as mx, mlx_lm, importlib.metadata as md
print("mlx-lm", md.version("mlx-lm"))
print("mlx", md.version("mlx"))
import mlx_lm.models.cache as c
print("make_prompt_cache present:", hasattr(c, "make_prompt_cache"))
print("cache module exports:", [n for n in dir(c) if not n.startswith("_")])
EOF
```

Expected: `make_prompt_cache present: True`. If it prints False, find the equivalent in the printed exports and note the substitution in `LEARNING-LOG.md` — every later task that says `make_prompt_cache` uses that name instead.

- [ ] **Step 5: Write .gitignore**

```gitignore
.venv/
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
.DS_Store
```

- [ ] **Step 6: Create the package skeleton**

```bash
mkdir -p eightserve/engine eightserve/server tests
touch eightserve/__init__.py eightserve/engine/__init__.py eightserve/server/__init__.py tests/__init__.py
```

- [ ] **Step 7: Create the three capture files**

`MEASUREMENTS.md`:

```markdown
# Measurements

Every number here has a date and the exact command that produced it.
Median of 3 runs after cooldown unless stated otherwise.

## Machine

- Apple M1, 8 GB unified memory, macOS 26.5
- Memory bandwidth: ~68 GB/s (the number every ceiling is computed from)

---
```

`SURPRISES.md`:

```markdown
# Surprises

One line whenever something confuses me or a prediction misses.
Post #1's "26 vs 860" moment came out of a file like this.

---
```

`LEARNING-LOG.md`:

```markdown
# Learning Log

What was built, why it was built that way, and what the source of
mlx-lm and nano-vllm actually revealed — written at the moment we
touched each piece.

`MEASUREMENTS.md` holds the numbers. `SURPRISES.md` holds the
confusions. This file holds the understanding.

---
```

- [ ] **Step 8: Write README.md**

```markdown
# eightserve

A readable LLM serving engine on an 8 GB M1 MacBook Air.

Not a fast one. A readable one — written to find out what
continuous batching and PagedAttention actually are, by building
them and measuring what they do.

Companion to [My laptop reads a gigabyte to write one word](#).

## What it is

An OpenAI-compatible HTTP server around Qwen2.5-1.5B-Instruct (4-bit)
on MLX, with a scheduler doing continuous batching and a block-based
KV cache allocator written by hand.

## Status

Under construction, in the open. See `MEASUREMENTS.md` for every
number, `LEARNING-LOG.md` for what each piece taught.

## Setup

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```
```

- [ ] **Step 9: Verify pytest runs**

Run: `.venv/bin/pytest -q`
Expected: `no tests ran` — an exit without collection errors is the pass condition.

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "chore: project skeleton, deps, capture files"
```

---

### Task 2: Sampler

The one module with no MLX state and no I/O, so it gets real unit tests. Written first while the 1 GB model downloads in the background.

**Files:**
- Create: `eightserve/engine/sampler.py`
- Test: `tests/test_sampler.py`

**Interfaces:**
- Consumes: nothing
- Produces: `sample(logits: mx.array, temperature: float = 0.0, top_p: float = 1.0) -> int`, taking a 1-D logits array of shape `(vocab,)` and returning a token id.

- [ ] **Step 1: Start the model download in the background**

It is ~1 GB and Task 4 blocks on it. Start it now, then keep working.

```bash
.venv/bin/python -c "
from mlx_lm import load
load('mlx-community/Qwen2.5-1.5B-Instruct-4bit')
print('model cached')
" &
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_sampler.py
import mlx.core as mx
import pytest

from eightserve.engine.sampler import sample


def test_greedy_picks_argmax():
    logits = mx.array([1.0, 5.0, 3.0, 2.0])
    assert sample(logits, temperature=0.0) == 1


def test_greedy_is_deterministic():
    logits = mx.array([0.1, 0.2, 9.0, 0.3])
    assert [sample(logits, temperature=0.0) for _ in range(5)] == [2] * 5


def test_temperature_zero_ignores_top_p():
    logits = mx.array([1.0, 5.0, 3.0])
    assert sample(logits, temperature=0.0, top_p=0.01) == 1


def test_top_p_excludes_low_probability_tokens():
    # Token 0 holds almost all the mass; nucleus at 0.5 can only contain it.
    logits = mx.array([20.0, 0.0, 0.0, 0.0])
    picks = {sample(logits, temperature=1.0, top_p=0.5) for _ in range(50)}
    assert picks == {0}


def test_top_p_one_can_reach_any_token():
    # A flat distribution with the full nucleus should eventually pick
    # something other than index 0.
    logits = mx.zeros((8,))
    picks = {sample(logits, temperature=1.0, top_p=1.0) for _ in range(200)}
    assert len(picks) > 1


def test_rejects_non_1d_logits():
    with pytest.raises(ValueError):
        sample(mx.zeros((1, 4)), temperature=0.0)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_sampler.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'eightserve.engine.sampler'`

- [ ] **Step 4: Implement the sampler**

```python
# eightserve/engine/sampler.py
"""Token sampling. Pure functions over a logits vector — no model state."""

import mlx.core as mx


def sample(logits: mx.array, temperature: float = 0.0, top_p: float = 1.0) -> int:
    """Pick the next token id from a 1-D logits array of shape (vocab,).

    temperature == 0.0 means greedy (argmax), which is what every
    measurement run uses: sampling noise would make two runs of the same
    prompt incomparable.
    """
    if logits.ndim != 1:
        raise ValueError(f"expected 1-D logits of shape (vocab,), got {logits.shape}")

    if temperature == 0.0:
        return int(mx.argmax(logits).item())

    scaled = logits * (1.0 / temperature)

    if top_p >= 1.0:
        return int(mx.random.categorical(scaled).item())

    probs = mx.softmax(scaled, axis=-1)
    order = mx.argsort(-probs)
    ordered = probs[order]
    cumulative = mx.cumsum(ordered)

    # Keep every token up to and including the one that crosses top_p, so
    # the nucleus is never empty even when one token holds all the mass.
    keep = cumulative - ordered < top_p
    filtered = mx.where(keep, ordered, mx.zeros_like(ordered))

    choice = mx.random.categorical(mx.log(filtered + 1e-10))
    return int(order[choice].item())
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_sampler.py -q`
Expected: 6 passed

- [ ] **Step 6: Note the design decision in LEARNING-LOG.md**

Append:

```markdown
## Sampler — why greedy is the default

`temperature=0.0` is the default and every measurement run uses it. With
sampling on, two runs of the same prompt produce different token counts and
different text, so tok/s numbers stop being comparable. Determinism is a
measurement requirement here, not a quality preference.

The top-p filter keeps tokens while `cumulative - ordered < top_p` rather
than `cumulative < top_p`. The difference matters when a single token holds
more probability than top_p: the naive form keeps nothing and the sample is
undefined. This form always keeps at least the top token.
```

- [ ] **Step 7: Commit**

```bash
git add eightserve/engine/sampler.py tests/test_sampler.py LEARNING-LOG.md
git commit -m "feat: sampler with greedy, temperature and top-p"
```

---

### Task 3: Sequence

**Files:**
- Create: `eightserve/engine/sequence.py`
- Test: `tests/test_sequence.py`

**Interfaces:**
- Consumes: nothing
- Produces: `SeqStatus` enum (`WAITING`, `RUNNING`, `FINISHED`), and `Sequence` with fields `id: str`, `prompt_tokens: list[int]`, `generated: list[int]`, `status: SeqStatus`, `max_tokens: int`, `temperature: float`, `top_p: float`, `stop_reason: str | None`, and methods `append(token_id)`, `should_stop(eos_id) -> bool`, `all_tokens() -> list[int]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sequence.py
from eightserve.engine.sequence import Sequence, SeqStatus


def make_seq(**kw):
    defaults = dict(id="s1", prompt_tokens=[1, 2, 3], max_tokens=4)
    return Sequence(**{**defaults, **kw})


def test_starts_waiting():
    assert make_seq().status is SeqStatus.WAITING


def test_append_accumulates_generated_tokens():
    seq = make_seq()
    seq.append(10)
    seq.append(11)
    assert seq.generated == [10, 11]


def test_all_tokens_is_prompt_plus_generated():
    seq = make_seq()
    seq.append(10)
    assert seq.all_tokens() == [1, 2, 3, 10]


def test_stops_at_max_tokens():
    seq = make_seq(max_tokens=2)
    seq.append(10)
    assert not seq.should_stop(eos_id=999)
    seq.append(11)
    assert seq.should_stop(eos_id=999)
    assert seq.stop_reason == "length"


def test_stops_on_eos():
    seq = make_seq(max_tokens=100)
    seq.append(999)
    assert seq.should_stop(eos_id=999)
    assert seq.stop_reason == "stop"


def test_eos_beats_length_when_both_apply():
    seq = make_seq(max_tokens=1)
    seq.append(999)
    assert seq.should_stop(eos_id=999)
    assert seq.stop_reason == "stop"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_sequence.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'eightserve.engine.sequence'`

- [ ] **Step 3: Implement**

```python
# eightserve/engine/sequence.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_sequence.py -q`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add eightserve/engine/sequence.py tests/test_sequence.py
git commit -m "feat: sequence state with stop conditions"
```

---

### Task 4: Runner — the decode loop we own

The only module that imports MLX model machinery. This is the task the whole approach decision was about.

**Files:**
- Create: `eightserve/engine/runner.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: nothing
- Produces: `Runner` with `Runner.load(model_id: str = DEFAULT_MODEL) -> Runner`, and instance members `model`, `tokenizer`, `eos_id: int`, `encode(text) -> list[int]`, `format_prompt(instruction: str) -> str`, `new_cache() -> list`, `prefill(token_ids: list[int], cache) -> mx.array`, `decode_step(token_id: int, cache) -> mx.array`. Both `prefill` and `decode_step` return **1-D logits of shape `(vocab,)`** for the next position, already evaluated. `DEFAULT_MODEL` is the module-level model id string.

- [ ] **Step 1: Confirm the model finished downloading**

```bash
.venv/bin/python -c "
from mlx_lm import load
m, t = load('mlx-community/Qwen2.5-1.5B-Instruct-4bit')
print('loaded ok')
"
```

Expected: `loaded ok`. If it is still downloading, wait — Task 5 needs it too.

- [ ] **Step 2: Write the failing tests**

These load the real model, so they carry the `slow` marker. MLX is not mocked: a mocked forward pass would pass while the real path is broken, which is the opposite of useful.

```python
# tests/test_runner.py
import mlx.core as mx
import pytest

from eightserve.engine.runner import Runner

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def runner():
    return Runner.load()


def test_prefill_returns_1d_logits(runner):
    ids = runner.encode("The capital of France is")
    logits = runner.prefill(ids, runner.new_cache())
    assert logits.ndim == 1
    assert logits.shape[0] > 1000  # vocab, not sequence length


def test_decode_step_returns_1d_logits(runner):
    cache = runner.new_cache()
    ids = runner.encode("The capital of France is")
    first = runner.prefill(ids, cache)
    nxt = int(mx.argmax(first).item())
    logits = runner.decode_step(nxt, cache)
    assert logits.shape == first.shape


def test_greedy_continuation_is_coherent(runner):
    """The real end-to-end check: our own loop, not mlx-lm's."""
    cache = runner.new_cache()
    ids = runner.encode("The capital of France is")
    logits = runner.prefill(ids, cache)
    out = []
    for _ in range(4):
        tok = int(mx.argmax(logits).item())
        out.append(tok)
        logits = runner.decode_step(tok, cache)
    assert "Paris" in runner.tokenizer.decode(out)


def test_fresh_cache_reproduces_the_same_tokens(runner):
    """Two independent caches, same prompt, greedy -> identical output.
    Catches cache state leaking between sequences, which is exactly the
    bug that would silently corrupt batched decoding in Session 4."""
    def run():
        cache = runner.new_cache()
        ids = runner.encode("Count: 1 2 3")
        logits = runner.prefill(ids, cache)
        out = []
        for _ in range(5):
            tok = int(mx.argmax(logits).item())
            out.append(tok)
            logits = runner.decode_step(tok, cache)
        return out

    assert run() == run()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_runner.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'eightserve.engine.runner'`

- [ ] **Step 4: Implement the runner**

```python
# eightserve/engine/runner.py
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_runner.py -q`
Expected: 4 passed. First run is slow — the model loads once per module.

If `test_greedy_continuation_is_coherent` fails with plausible-but-wrong text, the chat template is the first suspect: check whether `encode` is double-applying special tokens by printing `runner.encode(runner.format_prompt("hi"))` and comparing against `tokenizer.apply_chat_template(..., tokenize=True)`. Record the finding in `SURPRISES.md`.

- [ ] **Step 6: Write up what the mlx-lm source showed**

Append to `LEARNING-LOG.md`:

```markdown
## Runner — what mlx-lm actually gives you

`mlx_lm.load()` returns `(model, tokenizer)`. The model is a plain callable:
`model(ids, cache=cache)` returns logits of shape `(batch, seq_len, vocab)`.
That is the entire interface the generation loop needs — `generate_step` in
`mlx_lm/generate.py` is a convenience wrapper around exactly this call plus
sampling, and we replace it with our own thirty lines.

`make_prompt_cache(model)` returns a **list with one cache object per layer**,
not a single object. Each one holds that layer's K and V and grows as tokens
are appended. This is the thing Session 7 replaces with a block-based
allocator, and the list-per-layer shape is why the block table is per-layer too.

**Prefill and decode are the same call with a different sequence length.**
Prefill passes N tokens and gets N positions of logits back; we keep only
the last. Decode passes 1 token. Nothing else differs — no separate code
path, no special mode. The two phases from post #1 turn out to be one
function with a different input shape, which explains why prefill is
compute-bound (lots of tokens, one weight read) and decode is
memory-bound (one token, one weight read).

**MLX is lazy.** `model(...)` returns instantly and computes nothing. Without
`mx.eval(out)` inside prefill and decode_step, the timers in the CLI would
measure graph construction and the whole first measurement would be fiction.
```

- [ ] **Step 7: Commit**

```bash
git add eightserve/engine/runner.py tests/test_runner.py LEARNING-LOG.md
git commit -m "feat: runner owning prefill and decode step"
```

---

### Task 5: CLI with timing — records M1

**Files:**
- Create: `eightserve/prompts.py`, `eightserve/cli.py`
- Test: `tests/test_prompts.py`
- Modify: `MEASUREMENTS.md`

**Interfaces:**
- Consumes: `Runner`, `sample`
- Produces: `PROMPTS: dict[str, str]` with keys `short`, `medium`, `long`; `generate(runner, instruction, max_tokens, temperature, top_p) -> GenResult` where `GenResult` is a dataclass with `text: str`, `prompt_tokens: int`, `generated_tokens: int`, `ttft_s: float`, `prefill_s: float`, `decode_s: float`, `decode_tps: float`, `prefill_tps: float`, `peak_rss_gb: float`.

- [ ] **Step 1: Write the failing test for the frozen prompt set**

The test exists to make the freeze mechanical rather than a good intention. If someone edits a prompt, this fails and they have to notice.

```python
# tests/test_prompts.py
from eightserve.prompts import PROMPTS


def test_prompt_set_has_three_sizes():
    assert set(PROMPTS) == {"short", "medium", "long"}


def test_prompts_are_frozen():
    """These strings are locked for the life of the project. Changing one
    invalidates every measurement recorded before the change. If this test
    fails, the fix is to revert the prompt, not to update the test."""
    assert PROMPTS["short"].startswith("Write a haiku about")
    assert len(PROMPTS["short"]) == 44
    assert len(PROMPTS["medium"]) == 125
    assert len(PROMPTS["long"]) == 547
```

The three lengths (44, 125, 547) are the real values for the strings written in Step 2. If a length assertion fails, the prompt was edited — revert the prompt.

- [ ] **Step 2: Write the frozen prompt set**

```python
# eightserve/prompts.py
"""The frozen prompt set. Fixed 2026-08-19, never changed.

Changing any string here makes every measurement recorded before the
change incomparable to every measurement after it. That is the whole
reason this is a module and not an argument.
"""

PROMPTS = {
    "short": "Write a haiku about a laptop fan that isn't.",
    "medium": (
        "Explain in one paragraph why generating a single token from a "
        "language model requires reading the entire model out of memory."
    ),
    "long": (
        "Write a detailed essay of several paragraphs about the memory "
        "hierarchy of a modern computer. Cover registers, the cache "
        "levels, main memory, and storage. For each level, explain its "
        "approximate size, its approximate latency, and the reason that "
        "level exists at all rather than being merged into its neighbour. "
        "Then explain what the phrase 'memory bandwidth bound' means for "
        "a program whose inner loop reads more bytes than it performs "
        "arithmetic operations, and why adding faster arithmetic units to "
        "such a program produces no speedup whatsoever."
    ),
}
```

- [ ] **Step 3: Run the tests**

Run: `.venv/bin/pytest tests/test_prompts.py -q`
Expected: 2 passed

- [ ] **Step 4: Write the CLI**

```python
# eightserve/cli.py
"""Run the loop in-process and print what it cost.

Every timer here brackets an mx.eval() inside the runner, so the numbers
describe work actually done rather than graphs actually built.
"""

import argparse
import time
from dataclasses import dataclass, asdict

import psutil

from eightserve.engine.runner import Runner, DEFAULT_MODEL
from eightserve.engine.sampler import sample
from eightserve.prompts import PROMPTS


@dataclass
class GenResult:
    text: str
    prompt_tokens: int
    generated_tokens: int
    ttft_s: float
    prefill_s: float
    decode_s: float
    decode_tps: float
    prefill_tps: float
    peak_rss_gb: float


def generate(runner, instruction, max_tokens=128, temperature=0.0, top_p=1.0,
             on_text=None) -> GenResult:
    proc = psutil.Process()
    ids = runner.encode(runner.format_prompt(instruction))

    detok = runner.tokenizer.detokenizer
    detok.reset()

    cache = runner.new_cache()

    t0 = time.perf_counter()
    logits = runner.prefill(ids, cache)
    prefill_s = time.perf_counter() - t0

    pieces = []
    generated = 0
    ttft_s = None
    peak_rss = proc.memory_info().rss
    t_decode_start = time.perf_counter()

    for _ in range(max_tokens):
        tok = sample(logits, temperature=temperature, top_p=top_p)
        if tok == runner.eos_id:
            break

        if ttft_s is None:
            # First token is prefill plus one decode step — this is what a
            # user experiences as the pause before text appears.
            ttft_s = time.perf_counter() - t0

        detok.add_token(tok)
        segment = detok.last_segment
        if segment:
            pieces.append(segment)
            if on_text:
                on_text(segment)

        generated += 1
        peak_rss = max(peak_rss, proc.memory_info().rss)
        logits = runner.decode_step(tok, cache)

    detok.finalize()
    if detok.last_segment:
        pieces.append(detok.last_segment)
        if on_text:
            on_text(detok.last_segment)

    decode_s = time.perf_counter() - t_decode_start

    return GenResult(
        text="".join(pieces),
        prompt_tokens=len(ids),
        generated_tokens=generated,
        ttft_s=ttft_s or 0.0,
        prefill_s=prefill_s,
        decode_s=decode_s,
        decode_tps=generated / decode_s if decode_s else 0.0,
        prefill_tps=len(ids) / prefill_s if prefill_s else 0.0,
        peak_rss_gb=peak_rss / 1024**3,
    )


def main():
    ap = argparse.ArgumentParser(prog="eightserve.cli")
    ap.add_argument("--prompt", choices=sorted(PROMPTS), default="medium")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--quiet", action="store_true", help="stats only, no text")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    runner = Runner.load(args.model)
    instruction = PROMPTS[args.prompt]

    results = []
    for i in range(args.runs):
        printer = None if args.quiet else (lambda s: print(s, end="", flush=True))
        r = generate(runner, instruction, args.max_tokens,
                     args.temperature, args.top_p, on_text=printer)
        if not args.quiet:
            print()
        results.append(r)
        print(f"[run {i+1}] prompt={r.prompt_tokens}tok "
              f"generated={r.generated_tokens}tok "
              f"ttft={r.ttft_s*1000:.0f}ms "
              f"prefill={r.prefill_tps:.1f}tok/s "
              f"decode={r.decode_tps:.1f}tok/s "
              f"rss={r.peak_rss_gb:.2f}GB")

    if args.runs > 1:
        med = sorted(r.decode_tps for r in results)[len(results) // 2]
        print(f"[median of {args.runs}] decode={med:.1f}tok/s")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run it for real**

```bash
.venv/bin/python -m eightserve.cli --prompt medium --max-tokens 128
```

Expected: coherent text streaming to the terminal, then a stats line. If decode tok/s reads above ~70, something is wrong — that is above the machine's roofline ceiling, which means a timer is not bracketing real work. Investigate before recording anything.

- [ ] **Step 6: Take the M1 decode measurement**

Close Chrome first. The 8 GB budget is the whole story of this project and a browser eats half of it.

```bash
.venv/bin/python -m eightserve.cli --prompt medium --max-tokens 128 --quiet --runs 3
```

- [ ] **Step 7: Record M1's eightserve half in MEASUREMENTS.md**

Append, with the real numbers substituted:

```markdown
## M1 — baseline, 2026-08-19

Command:
```
.venv/bin/python -m eightserve.cli --prompt medium --max-tokens 128 --quiet --runs 3
```

| Quantity | Value |
|---|---|
| eightserve decode | __ tok/s (median of 3) |
| eightserve prefill | __ tok/s |
| TTFT (medium prompt) | __ ms |
| Peak RSS | __ GB |
```

- [ ] **Step 8: Commit**

```bash
git add eightserve/prompts.py eightserve/cli.py tests/test_prompts.py MEASUREMENTS.md
git commit -m "feat: cli with timing; record M1 decode baseline"
```

---

### Task 6: The comparison numbers — roofline and llama.cpp

No new code. This is the task that makes M1 mean something.

**Files:**
- Modify: `MEASUREMENTS.md`, `LEARNING-LOG.md`, `SURPRISES.md`

**Interfaces:**
- Consumes: the M1 table from Task 5
- Produces: nothing consumed by later tasks

- [ ] **Step 1: Measure the model's actual on-disk size**

The roofline ceiling is bandwidth divided by *bytes actually read per token*, so it needs the real size of the MLX 4-bit weights, not the GGUF's.

```bash
du -sh ~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-4bit
du -sm ~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-4bit
```

- [ ] **Step 2: Compute the ceiling**

```bash
.venv/bin/python -c "
bw = 68.0          # GB/s, M1 unified memory
size = REPLACE_ME  # GB, from step 1
print(f'roofline ceiling: {bw/size:.1f} tok/s')
"
```

- [ ] **Step 3: Re-run llama-bench today, on the same machine**

Post #1's 26 tok/s is not reused. A number measured weeks ago on a differently-loaded machine cannot be the "before" half of a before/after table.

```bash
llama-bench -m ~/models/qwen2.5-1.5b-instruct-q4_k_m.gguf -p 512 -n 128
```

- [ ] **Step 4: Complete the M1 table**

Append to the M1 section in `MEASUREMENTS.md`:

```markdown
| Roofline ceiling (68 GB/s ÷ __ GB) | __ tok/s |
| llama.cpp Q4_K_M (`llama-bench -p 512 -n 128`, today) | __ tok/s |
| llama.cpp Q4_K_M (post #1, weeks ago) | 26 tok/s |

Both engines run the same model at the same quantization on the same
machine on the same day. The gap between them is a fact about the two
programs, not about two measurement sessions.
```

- [ ] **Step 5: Write down the prediction and the miss**

Before looking hard at the numbers, write the prediction in `SURPRISES.md`, then the outcome. The gap is the content.

```markdown
- 2026-08-19: predicted eightserve would land within ~20% of llama.cpp,
  since both are memory-bound on the same weights. Actual: __. Reason
  I currently believe: __
```

- [ ] **Step 6: Explain the comparison in LEARNING-LOG.md**

```markdown
## M1 — why three numbers instead of one

A tok/s figure alone says nothing. It needs two neighbours:

- **The roofline ceiling** (bandwidth ÷ bytes per token) is what the machine
  physically permits. Coming in near it means the loop is fine and the
  hardware is the limit. Coming in far under it means there is real work to do.
- **llama.cpp on the same weights** is what a mature, heavily optimised
  implementation achieves on this exact machine. It sets the bar.

eightserve sitting between them is the expected and honest result. The
distance to the ceiling is the headroom; the distance to llama.cpp is the
cost of being readable instead of fast. Both distances are the post.
```

- [ ] **Step 7: Commit**

```bash
git add MEASUREMENTS.md LEARNING-LOG.md SURPRISES.md
git commit -m "docs: complete M1 with roofline ceiling and llama.cpp comparison"
```

---

### Task 7: SSE framing

**Files:**
- Create: `eightserve/server/sse.py`
- Test: `tests/test_sse.py`

**Interfaces:**
- Consumes: nothing
- Produces: `sse_frame(payload: dict) -> str`, `SSE_DONE: str`, `completion_chunk(seq_id: str, text: str, model: str, created: int) -> dict`, `completion_final(seq_id: str, finish_reason: str, model: str, created: int) -> dict`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sse.py
import json

from eightserve.server.sse import (
    SSE_DONE,
    completion_chunk,
    completion_final,
    sse_frame,
)


def test_frame_is_data_line_with_blank_line_terminator():
    assert sse_frame({"a": 1}) == 'data: {"a": 1}\n\n'


def test_frame_has_no_raw_newlines_in_payload():
    """A newline inside the JSON would end the SSE event early and split
    one message into two. json.dumps escapes it; this pins that down."""
    frame = sse_frame({"text": "line one\nline two"})
    assert frame.count("\n") == 2
    assert frame.endswith("\n\n")


def test_done_sentinel_is_literal():
    assert SSE_DONE == "data: [DONE]\n\n"


def test_completion_chunk_shape():
    c = completion_chunk("s1", "hello", model="m", created=1)
    assert c["object"] == "text_completion"
    assert c["choices"][0]["text"] == "hello"
    assert c["choices"][0]["finish_reason"] is None


def test_completion_final_carries_finish_reason():
    c = completion_final("s1", "length", model="m", created=1)
    assert c["choices"][0]["finish_reason"] == "length"
    assert c["choices"][0]["text"] == ""


def test_chunk_round_trips_through_a_frame():
    frame = sse_frame(completion_chunk("s1", "hi", model="m", created=1))
    payload = json.loads(frame[len("data: "):].strip())
    assert payload["choices"][0]["text"] == "hi"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_sse.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'eightserve.server.sse'`

- [ ] **Step 3: Implement**

```python
# eightserve/server/sse.py
"""Server-sent-event framing and OpenAI-shaped completion payloads.

Pure string and dict construction, so the wire format is testable without
a server, a client, or a model.
"""

import json

SSE_DONE = "data: [DONE]\n\n"


def sse_frame(payload: dict) -> str:
    """One SSE event. The blank line is the terminator — without it the
    client buffers forever waiting for the event to end."""
    return f"data: {json.dumps(payload)}\n\n"


def completion_chunk(seq_id: str, text: str, model: str, created: int) -> dict:
    return {
        "id": seq_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{"text": text, "index": 0, "logprobs": None,
                     "finish_reason": None}],
    }


def completion_final(seq_id: str, finish_reason: str, model: str,
                     created: int) -> dict:
    return {
        "id": seq_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{"text": "", "index": 0, "logprobs": None,
                     "finish_reason": finish_reason}],
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_sse.py -q`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add eightserve/server/sse.py tests/test_sse.py
git commit -m "feat: SSE framing and OpenAI completion payloads"
```

---

### Task 8: Engine thread

**Files:**
- Create: `eightserve/engine/engine.py`
- Test: `tests/test_engine.py`

**Interfaces:**
- Consumes: `Runner`, `Sequence`, `SeqStatus`, `sample`
- Produces: `Engine(runner)` with `start()`, `stop()`, `submit(seq: Sequence) -> None`. The engine pushes `("text", str)` tuples onto `seq.out_queue` as generation proceeds and exactly one `("done", finish_reason)` tuple at the end, always, including on error — where it first pushes `("error", message)`.

- [ ] **Step 1: Write the failing tests**

The engine is tested with a fake runner. This is not mocking MLX to avoid the model — it is testing the *threading and queue contract*, which has nothing to do with the model and would be untestably slow if it did.

```python
# tests/test_engine.py
import asyncio

import pytest

from eightserve.engine.engine import Engine
from eightserve.engine.sequence import Sequence, SeqStatus


class FakeRunner:
    """Emits tokens 100, 101, 102, then EOS. Deterministic, instant."""
    eos_id = 999

    def __init__(self):
        self.tokenizer = self
        self.detokenizer = self
        self._segments = []

    # --- runner surface ---
    def encode(self, text):
        return [1, 2, 3]

    def new_cache(self):
        return {"n": 0}

    def prefill(self, ids, cache):
        cache["n"] = 0
        return self._logits_for(100)

    def decode_step(self, token_id, cache):
        cache["n"] += 1
        nxt = [101, 102, 999][min(cache["n"] - 1, 2)]
        return self._logits_for(nxt)

    def _logits_for(self, token_id):
        import mlx.core as mx
        v = [0.0] * 1000
        v[token_id] = 10.0
        return mx.array(v)

    # --- detokenizer surface ---
    def reset(self):
        self._segments = []

    def add_token(self, t):
        self._segments.append(f"<{t}>")

    def finalize(self):
        pass

    @property
    def last_segment(self):
        return self._segments[-1] if self._segments else ""


async def drain(seq):
    out, reason = [], None
    while True:
        kind, value = await asyncio.wait_for(seq.out_queue.get(), timeout=10)
        if kind == "text":
            out.append(value)
        elif kind == "error":
            raise AssertionError(f"engine error: {value}")
        else:
            reason = value
            break
    return "".join(out), reason


async def test_engine_streams_tokens_to_the_queue():
    engine = Engine(FakeRunner())
    engine.start()
    try:
        seq = Sequence(id="s1", prompt_tokens=[1, 2], max_tokens=10,
                       out_queue=asyncio.Queue(),
                       loop=asyncio.get_running_loop())
        engine.submit(seq)
        text, reason = await drain(seq)
        assert text == "<100><101><102>"
        assert reason == "stop"
        assert seq.status is SeqStatus.FINISHED
    finally:
        engine.stop()


async def test_max_tokens_truncates_and_reports_length():
    engine = Engine(FakeRunner())
    engine.start()
    try:
        seq = Sequence(id="s2", prompt_tokens=[1], max_tokens=2,
                       out_queue=asyncio.Queue(),
                       loop=asyncio.get_running_loop())
        engine.submit(seq)
        text, reason = await drain(seq)
        assert text == "<100><101>"
        assert reason == "length"
    finally:
        engine.stop()


async def test_two_sequences_do_not_interleave_or_share_cache():
    engine = Engine(FakeRunner())
    engine.start()
    try:
        loop = asyncio.get_running_loop()
        seqs = [Sequence(id=f"s{i}", prompt_tokens=[1], max_tokens=10,
                         out_queue=asyncio.Queue(), loop=loop)
                for i in range(2)]
        for s in seqs:
            engine.submit(s)
        results = await asyncio.gather(*(drain(s) for s in seqs))
        assert [r[0] for r in results] == ["<100><101><102>"] * 2
    finally:
        engine.stop()


async def test_engine_reports_errors_instead_of_hanging():
    class Broken(FakeRunner):
        def prefill(self, ids, cache):
            raise RuntimeError("boom")

    engine = Engine(Broken())
    engine.start()
    try:
        seq = Sequence(id="s3", prompt_tokens=[1], max_tokens=4,
                       out_queue=asyncio.Queue(),
                       loop=asyncio.get_running_loop())
        engine.submit(seq)
        kind, value = await asyncio.wait_for(seq.out_queue.get(), timeout=10)
        assert kind == "error"
        assert "boom" in value
        kind, _ = await asyncio.wait_for(seq.out_queue.get(), timeout=10)
        assert kind == "done"
    finally:
        engine.stop()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_engine.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'eightserve.engine.engine'`

- [ ] **Step 3: Implement**

```python
# eightserve/engine/engine.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_engine.py -q`
Expected: 4 passed

Note the ordering inside the loop, which is the one subtle thing in this file. A token is emitted *before* the stop check, so a sequence ending at `max_tokens` still delivers its last token — checking first drops it, and the bug is invisible with generous `max_tokens`. EOS is excluded from emission explicitly, which is why `test_engine_streams_tokens_to_the_queue` expects three segments rather than four.

- [ ] **Step 5: Write up the threading decision**

Append to `LEARNING-LOG.md`:

```markdown
## Engine — why a thread and not just await

MLX calls are synchronous and block for the whole forward pass. Calling one
inside an async handler holds the event loop for that entire duration, so
every other connection — including ones with tokens ready to send — waits.
With one user this is invisible. With eight, which is the entire point of
the project, it destroys the result.

So: one dedicated thread owns the runner, and `loop.call_soon_threadsafe`
hands each token to the event loop that owns that request's queue. Handlers
only ever `await queue.get()`.

The batch size is 1 today. Nothing else about this file changes in Session 6
— the scheduler replaces the body of `_generate`, and the queue contract
above it stays as it is. Building this shape now costs nothing; skipping it
would cost a rewrite in the session that already carries the most risk.

The `("text", ...)` / `("done", ...)` / `("error", ...)` tuple protocol
exists so the handler can tell "the stream ended" from "the stream ended
badly" without inspecting engine state across a thread boundary. Exactly one
`done` is always sent, including on error, or the handler hangs forever.
```

- [ ] **Step 6: Commit**

```bash
git add eightserve/engine/engine.py tests/test_engine.py LEARNING-LOG.md
git commit -m "feat: engine thread with per-request queues"
```

---

### Task 9: HTTP server

**Files:**
- Create: `eightserve/server/app.py`
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: `Engine`, `Runner`, `Sequence`, `sse_frame`, `SSE_DONE`, `completion_chunk`, `completion_final`
- Produces: `create_app(runner) -> FastAPI` and module-level `app` built lazily from a real `Runner`. Endpoints: `GET /health`, `POST /v1/completions`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_app.py
import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from eightserve.server.app import create_app
from tests.test_engine import FakeRunner


@pytest.fixture
def client():
    app = create_app(FakeRunner())
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


async def test_health(client):
    async with client as c:
        r = await c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


async def test_completions_streams_sse_frames(client):
    async with client as c:
        async with c.stream("POST", "/v1/completions", json={
            "prompt": "hello", "max_tokens": 10, "stream": True,
        }) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
            body = "".join([chunk async for chunk in r.aiter_text()])

    assert body.endswith("data: [DONE]\n\n")
    assert "<100>" in body and "<102>" in body


async def test_completions_non_streaming_returns_one_json_body(client):
    async with client as c:
        r = await c.post("/v1/completions", json={
            "prompt": "hello", "max_tokens": 10, "stream": False,
        })
    assert r.status_code == 200
    payload = r.json()
    assert payload["choices"][0]["text"] == "<100><101><102>"
    assert payload["choices"][0]["finish_reason"] == "stop"


async def test_handler_does_not_block_the_event_loop(client):
    """The rule the whole design rests on: while a completion is in
    flight, other requests still get served."""
    async with client as c:
        async def slow_request():
            r = await c.post("/v1/completions",
                             json={"prompt": "hello", "max_tokens": 10,
                                   "stream": False})
            return r.status_code

        task = asyncio.create_task(slow_request())
        health = await c.get("/health")
        assert health.status_code == 200
        assert await task == 200
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_app.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'eightserve.server.app'`

- [ ] **Step 3: Implement**

```python
# eightserve/server/app.py
"""OpenAI-compatible HTTP surface.

The handler never touches MLX. It creates a Sequence, hands it to the
engine thread, and awaits tokens on an asyncio.Queue. Compatibility with
/v1/completions is deliberate: any existing client can point at this.
"""

import time
import uuid
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from eightserve.engine.engine import Engine
from eightserve.engine.runner import Runner, DEFAULT_MODEL
from eightserve.engine.sequence import Sequence
from eightserve.server.sse import (
    SSE_DONE,
    completion_chunk,
    completion_final,
    sse_frame,
)


class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    stream: bool = True
    model: str = DEFAULT_MODEL


def create_app(runner) -> FastAPI:
    app = FastAPI(title="eightserve")
    engine = Engine(runner)

    @app.on_event("startup")
    def _start():
        engine.start()

    @app.on_event("shutdown")
    def _stop():
        engine.stop()

    def _new_sequence(req: CompletionRequest) -> Sequence:
        import asyncio
        return Sequence(
            id=f"cmpl-{uuid.uuid4().hex[:12]}",
            prompt_tokens=runner.tokenizer.encode(req.prompt),
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            out_queue=asyncio.Queue(),
            loop=asyncio.get_running_loop(),
        )

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": DEFAULT_MODEL}

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        seq = _new_sequence(req)
        engine.submit(seq)
        created = int(time.time())

        if req.stream:
            return StreamingResponse(
                _stream(seq, req.model, created),
                media_type="text/event-stream",
            )

        pieces, reason = [], "stop"
        while True:
            kind, value = await seq.out_queue.get()
            if kind == "text":
                pieces.append(value)
            elif kind == "error":
                reason = "error"
            else:
                reason = value if reason != "error" else "error"
                break

        return {
            "id": seq.id,
            "object": "text_completion",
            "created": created,
            "model": req.model,
            "choices": [{"text": "".join(pieces), "index": 0,
                         "logprobs": None, "finish_reason": reason}],
        }

    return app


async def _stream(seq: Sequence, model: str, created: int) -> AsyncIterator[str]:
    while True:
        kind, value = await seq.out_queue.get()
        if kind == "text":
            yield sse_frame(completion_chunk(seq.id, value, model, created))
        elif kind == "error":
            yield sse_frame(completion_final(seq.id, "error", model, created))
        else:
            yield sse_frame(completion_final(seq.id, value, model, created))
            yield SSE_DONE
            return


app = None


def get_app() -> FastAPI:
    """Entry point for `uvicorn eightserve.server.app:get_app --factory`.
    Loading the model here rather than at import keeps the test suite from
    pulling 1 GB off disk."""
    return create_app(Runner.load())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_app.py -q`
Expected: 4 passed

If FastAPI emits a `DeprecationWarning` about `on_event`, that is fine for now — swapping to a lifespan context manager is a two-line change and does not belong in the session that is measuring throughput.

- [ ] **Step 5: Run the real server and see tokens over curl**

```bash
.venv/bin/uvicorn eightserve.server.app:get_app --factory --port 8000
```

In a second terminal:

```bash
curl -N -s http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain the KV cache in two sentences.","max_tokens":64}'
```

Expected: `data: {...}` frames arriving progressively, ending with `data: [DONE]`. If everything arrives at once, buffering is in play — confirm `curl -N` was used before suspecting the server.

- [ ] **Step 6: Commit**

```bash
git add eightserve/server/app.py tests/test_app.py
git commit -m "feat: OpenAI-compatible completions endpoint with SSE"
```

---

### Task 10: Measure the server layer — records M2

**Files:**
- Create: `scripts/measure_http.py`
- Modify: `MEASUREMENTS.md`, `LEARNING-LOG.md`

**Interfaces:**
- Consumes: the running server, `PROMPTS`, `Runner.format_prompt`
- Produces: nothing consumed by later tasks. This is *not* `bench/harness.py` — that arrives in Session 3 and drives N concurrent clients. This script measures one request.

- [ ] **Step 1: Write the script**

```python
# scripts/measure_http.py
"""One request over HTTP, timed the same way the CLI times in-process.

The difference between this number and the CLI's number is what the
server layer costs. Same prompt, same chat template, same max_tokens, or
the comparison means nothing.
"""

import argparse
import json
import statistics
import time

import httpx

from eightserve.engine.runner import Runner
from eightserve.prompts import PROMPTS


def one_request(url: str, prompt: str, max_tokens: int) -> tuple[float, float, int]:
    t0 = time.perf_counter()
    ttft = None
    tokens = 0

    with httpx.Client(timeout=300) as client:
        with client.stream("POST", url, json={
            "prompt": prompt, "max_tokens": max_tokens, "stream": True,
        }) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                body = line[len("data: "):]
                if body == "[DONE]":
                    break
                payload = json.loads(body)
                text = payload["choices"][0]["text"]
                if text:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    tokens += 1

    total = time.perf_counter() - t0
    return ttft or 0.0, total, tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/v1/completions")
    ap.add_argument("--prompt", choices=sorted(PROMPTS), default="medium")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    # The chat template is applied client-side so the model sees exactly the
    # same tokens the CLI feeds it. /v1/completions stays a raw-prompt
    # endpoint, which is what OpenAI compatibility means.
    runner = Runner.load()
    prompt = runner.format_prompt(PROMPTS[args.prompt])

    ttfts, tpss = [], []
    for i in range(args.runs):
        ttft, total, tokens = one_request(args.url, prompt, args.max_tokens)
        decode_tps = (tokens - 1) / (total - ttft) if total > ttft else 0.0
        ttfts.append(ttft)
        tpss.append(decode_tps)
        print(f"[run {i+1}] ttft={ttft*1000:.0f}ms "
              f"segments={tokens} decode={decode_tps:.1f}tok/s")

    print(f"[median of {args.runs}] ttft={statistics.median(ttfts)*1000:.0f}ms "
          f"decode={statistics.median(tpss):.1f}tok/s")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it against the live server**

Server in one terminal (Chrome closed), this in another:

```bash
mkdir -p scripts
.venv/bin/python scripts/measure_http.py --prompt medium --max-tokens 128 --runs 3
```

Note: `segments` counts SSE frames carrying text, which is not exactly token count — the detokenizer emits a segment only when it has complete text to emit, so a multi-token character produces fewer segments than tokens. Record the number honestly as segments/s and note the caveat rather than claiming it is tok/s.

- [ ] **Step 3: Record M2**

Append to `MEASUREMENTS.md`:

```markdown
## M2 — what the server layer cost, 2026-08-19

Commands:
```
.venv/bin/python -m eightserve.cli --prompt medium --max-tokens 128 --quiet --runs 3
.venv/bin/python scripts/measure_http.py --prompt medium --max-tokens 128 --runs 3
```

| Path | TTFT | Decode |
|---|---|---|
| In-process (CLI) | __ ms | __ tok/s |
| Over HTTP + SSE | __ ms | __ segments/s |
| **Delta** | __ ms | __ |

Both paths run the identical templated prompt at the same max_tokens on a
machine with the browser closed. The HTTP figure counts SSE text segments,
not tokens — the detokenizer emits a segment only when it holds complete
text, so segments ≤ tokens.
```

- [ ] **Step 4: Write the session-2 conclusion**

Append to `LEARNING-LOG.md`:

```markdown
## M2 — what a server layer actually costs

The prediction going in: HTTP adds fixed overhead to TTFT (connection,
JSON parsing, one scheduling hop through the queue) and close to nothing to
decode throughput, because decode is bound by weight reads and the network
hop happens while the GPU is already busy on the next token.

Measured: __

If decode over HTTP came out materially slower, the first suspect is the
handler blocking somewhere it should be awaiting, not the network — on
localhost there is barely any network to blame.
```

- [ ] **Step 5: Run the whole suite**

```bash
.venv/bin/pytest -q -m "not slow"
.venv/bin/pytest -q -m slow
```

Expected: everything passes. The fast suite should finish in seconds; that speed is why the sampler, sequence, SSE and engine tests avoid the model.

- [ ] **Step 6: Commit**

```bash
git add scripts/measure_http.py MEASUREMENTS.md LEARNING-LOG.md
git commit -m "feat: HTTP measurement script; record M2"
```

---

## Done when

- `.venv/bin/pytest -q` passes, fast tests and slow tests both
- `MEASUREMENTS.md` contains a complete M1 table (eightserve decode, eightserve prefill, roofline ceiling, llama.cpp today) and a complete M2 table
- `LEARNING-LOG.md` explains the runner, the engine thread, and both measurements
- `SURPRISES.md` has at least one entry — if nothing surprised you across two sessions, the predictions were not written down before the measurements
- `curl -N` against the running server shows tokens arriving progressively

## Not in this plan

Sessions 3–10, each of which gets its own plan: the N-client benchmark harness (3), static batching with padding and attention masks (4–5), continuous batching (6), the paged block manager (7–8), prefix sharing (9), and the write-up (10).
