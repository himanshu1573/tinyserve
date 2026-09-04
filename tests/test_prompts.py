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
