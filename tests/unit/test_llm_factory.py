import pytest
from niftsy.llm.factory import resolve_provider, build_llm_backend
from niftsy.exceptions import NiftsyError


@pytest.mark.parametrize("model,expected", [
    ("gemini-2.5-flash", "gemini"),
    ("gpt-4o-mini", "openai"),
    ("Qwen/Qwen2.5-14B-Instruct", "local"),
])
def test_auto_resolve(model, expected):
    assert resolve_provider(model, "auto") == expected


def test_missing_extra_gives_clear_error(monkeypatch):
    # simulate google-genai not installed
    import builtins
    real_import = builtins.__import__
    def fake_import(name, *a, **k):
        if name.startswith("google"):
            raise ImportError("no module")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(NiftsyError, match="niftsy\\[gemini\\]"):
        build_llm_backend(model="gemini-2.5-flash", provider="gemini")
