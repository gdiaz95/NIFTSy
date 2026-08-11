import pytest
from niftsy.llm.factory import resolve_provider


@pytest.mark.parametrize("model,expected", [
    ("gemini-2.5-flash", "gemini"),
    ("gpt-4o-mini", "openai"),
    ("Qwen/Qwen2.5-14B-Instruct", "local"),
])
def test_auto_resolve(model, expected):
    assert resolve_provider(model, "auto") == expected
