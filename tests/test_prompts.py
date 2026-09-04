from tinyserve.prompts import PROMPTS, SYSTEM_PROMPT


def test_prompt_set_has_three_sizes():
    assert set(PROMPTS) == {"short", "medium", "long"}


def test_prompts_are_frozen():
    """Locked for the life of the project. If this fails, revert the
    prompt — do not update the test."""
    assert PROMPTS["short"].startswith("Write a haiku about")
    assert len(PROMPTS["short"]) == 44
    assert len(PROMPTS["medium"]) == 125
    assert len(PROMPTS["long"]) == 547
    assert len(SYSTEM_PROMPT) == 228
