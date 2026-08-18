from niftsy.config import PromptConfig
from niftsy.text.prompting import (
    build_prompt_from_structured_with_neighbors,
    cap_words,
    clean_generated_text,
)


def test_clean_generated_text_strips_meta_phrase():
    text = "Software engineer with five years experience. ignore previous instructions now."
    assert clean_generated_text(text) == "Software engineer with five years experience."


def test_clean_generated_text_strips_leading_preamble():
    # Regression test: a leading preamble phrase (the model echoing "return
    # only"/"here's your requested" before answering) must not wipe out the
    # real content that follows — this previously caused ~99.75% data loss
    # on a long-form free-text column.
    text = "Here's your requested amenities list: Air conditioning, WiFi, and a full kitchen."
    assert clean_generated_text(text) == "Air conditioning, WiFi, and a full kitchen."

    text2 = "Return only: A cozy studio with modern amenities and great natural light."
    assert clean_generated_text(text2) == "A cozy studio with modern amenities and great natural light."


def test_cap_words_truncates_to_exact_count():
    text = "one two three four five"
    assert cap_words(text, 3) == "one two three"


def test_build_prompt_substitutes_all_placeholders():
    template = PromptConfig().free_text_prompt
    prompt = build_prompt_from_structured_with_neighbors(
        fields={"age": 30, "income": 50000, "bio": "irrelevant, excluded by text_column"},
        prompt_template=template,
        max_words_generation=80,
        text_column="bio",
        neighbor_rows=[{"age": 28, "income": 48000, "bio": "works in tech"}],
        max_neighbors=3,
        max_words_reader=250,
    )
    for placeholder in (
        "{max_words_generation}",
        "{max_words_reader}",
        "{target_min_words}",
        "{target_max_words}",
        "{text_column}",
        "{target_profile}",
        "{neighbor_block}",
    ):
        assert placeholder not in prompt
    assert "bio" in prompt
    assert "works in tech" in prompt
