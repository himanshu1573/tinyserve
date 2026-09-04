"""The correctness contract of the whole project: every batched and paged
path must produce exactly the tokens the single-sequence loop produces.

If a batch of eight users got different text than one user would, the
headline throughput number would be measuring a different program.
"""

import mlx.core as mx
import pytest

from tinyserve.config import EngineConfig
from tinyserve.engine.backends import make_backend
from tinyserve.engine.runner import Runner
from tinyserve.engine.scheduler import Scheduler
from tinyserve.engine.sequence import Sequence
from tinyserve.prompts import PROMPTS, SYSTEM_PROMPT

pytestmark = pytest.mark.slow

N_TOKENS = 12


@pytest.fixture(scope="module")
def runner():
    return Runner.load()


def reference(runner, ids, n=N_TOKENS):
    """Session 1's loop: prefill + decode_step on a contiguous cache.
    Returns the greedy tokens plus, per position, the runner-up token and
    the top-2 logit margin, so a batched path can be judged fairly."""
    cache = runner.new_cache()
    logits = runner.prefill(ids, cache)
    out, runner_up, margin = [], [], []
    for _ in range(n):
        top2 = mx.argsort(-logits)[:2].tolist()
        l32 = logits.astype(mx.float32)
        out.append(top2[0])
        runner_up.append(top2[1])
        margin.append((l32[top2[0]] - l32[top2[1]]).item())
        if top2[0] in runner.eos_ids:
            break
        logits = runner.decode_step(top2[0], cache)
    return out, runner_up, margin


# fp16 greedy decoding is only deterministic up to ties. The frozen haiku
# prompt has an exact 0.0 margin at its fourth token, and a different (but
# equally valid) reduction order flips it. A batched path is judged to
# match if it reproduces the reference exactly, or diverges only at a
# near-tie by picking the runner-up — after which the contexts differ and
# further comparison is meaningless.
TIE_MARGIN = 0.25


def assert_matches(got, ref):
    tokens, runner_up, margin = ref
    for i, (g, e) in enumerate(zip(got, tokens)):
        if g == e:
            continue
        assert g == runner_up[i] and margin[i] < TIE_MARGIN, (
            f"diverged at {i}: got {g}, expected {e} (runner-up {runner_up[i]}, "
            f"margin {margin[i]:.3f})")
        return
    assert len(got) == len(tokens)


def scheduled(runner, prompts, *, kv_backend, budget_bytes=64 * 1024 * 1024,
              block_size=16, max_batch=8, prefix_caching=True, n=N_TOKENS):
    backend = make_backend(kv_backend, runner.kv_spec, budget_bytes,
                           block_size=block_size, prefix_caching=prefix_caching)
    tokens: dict[str, list[int]] = {}
    sched = Scheduler(runner, backend, max_batch_size=max_batch,
                      on_token=lambda s, t: tokens.setdefault(s.id, []).append(t))
    seqs = [Sequence(id=f"s{i}", prompt_tokens=ids, max_tokens=n)
            for i, ids in enumerate(prompts)]
    for s in seqs:
        sched.add(s)
    for _ in range(10_000):
        if not sched.has_work():
            break
        sched.step()
    assert not sched.has_work()
    return [tokens[s.id] for s in seqs], sched


@pytest.fixture(scope="module")
def prompts(runner):
    return [runner.encode(runner.format_prompt(PROMPTS[k]))
            for k in ("short", "medium", "long", "short")]


@pytest.fixture(scope="module")
def expected(runner, prompts):
    return [reference(runner, ids) for ids in prompts]


@pytest.mark.parametrize("kv_backend", ["paged", "padded"])
def test_batched_matches_single_sequence(runner, prompts, expected, kv_backend):
    got, sched = scheduled(runner, prompts, kv_backend=kv_backend)
    for g, e in zip(got, expected):
        assert_matches(g, e)
    assert sched.num_preemptions == 0


def test_paged_with_forced_preemption_matches(runner, prompts):
    # A pool of exactly 7 blocks. medium (52 tokens, 4 blocks) is admitted
    # first; short (41 tokens) needs 3 <= 3 free, so it joins, sharing the
    # chat-template block: 6 used, 1 free. short grows into the last block
    # at token 49; medium needs a 5th block at token 65 and finds none, so
    # the newest sequence (short) is preempted, waits, and is re-prefilled
    # with a warm prefix cache. Output must not change.
    n = 16
    seven_blocks = 7 * 16 * runner.kv_spec.bytes_per_token
    ps = [prompts[1], prompts[0]]
    es = [reference(runner, ids, n) for ids in ps]
    got, sched = scheduled(runner, ps, kv_backend="paged", budget_bytes=seven_blocks, n=n)
    for g, e in zip(got, es):
        assert_matches(g, e)
    assert sched.num_preemptions > 0
    assert sched.backend.bm.cache_hits > 1          # re-admission hit its old blocks
    assert sched.backend.bm.num_free == sched.backend.bm.num_blocks


def test_prefix_sharing_matches_and_hits(runner):
    prompts = [runner.encode(runner.format_prompt(PROMPTS[k], system=SYSTEM_PROMPT))
               for k in ("short", "medium", "long")]
    expected = [reference(runner, ids) for ids in prompts]
    got, sched = scheduled(runner, prompts, kv_backend="paged", max_batch=1)
    for g, e in zip(got, expected):
        assert_matches(g, e)
    assert sched.backend.bm.cache_hits > 0


def test_prefix_sharing_off_also_matches(runner, prompts, expected):
    got, sched = scheduled(runner, prompts, kv_backend="paged", prefix_caching=False)
    for g, e in zip(got, expected):
        assert_matches(g, e)
    assert sched.backend.bm.cache_hits == 0
