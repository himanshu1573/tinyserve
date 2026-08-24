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
