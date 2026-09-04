"""The frozen prompt set. Fixed on day one, never changed.

Changing any string here makes every measurement recorded before the
change incomparable to every measurement after it. That is the whole
reason this is a module and not a command-line argument.
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

# A shared system prompt for the prefix-sharing measurement (M7). Every
# one of N users sends this same preamble followed by their own prompt.
SYSTEM_PROMPT = (
    "You are a concise assistant running on a laptop with eight gigabytes "
    "of memory. Answer in plain English, avoid lists unless asked, and "
    "never apologise. When a question is about computers, prefer concrete "
    "numbers over adjectives."
)
